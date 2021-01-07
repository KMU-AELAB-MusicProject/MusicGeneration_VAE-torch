import os
import random
import numpy as np

from tqdm import tqdm
import shutil
import random

import logging

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from graph.model_with_gan import Model
from graph.phrase_encoder import PhraseModel
from graph.bar_discriminator import BarDiscriminator
from graph.bar_discriminator_with_feature import BarFeatureDiscriminator
from graph.z_discriminator import PhraseZDiscriminator, BarZDiscriminator
from graph.loss.bar_loss import Loss, DLoss

from data.bar_dataset import NoteDataset

from metrics import AverageMeter
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class BarGen(object):
    def __init__(self, config):
        self.config = config
        self.flag_gan = False
        self.train_count = 0

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size + 2
        
        self.logger = self.set_logger()

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        # define models ( generator and discriminator)
        self.generator = Model()
        self.discriminator = BarDiscriminator()
        self.discriminator_feature = BarFeatureDiscriminator()
        self.z_discriminator_phrase = PhraseZDiscriminator()
        self.z_discriminator_bar = BarZDiscriminator()

        # define loss
        self.loss_generator = Loss().cuda()
        self.loss_disc = DLoss().cuda()
        self.loss_feature_disc = DLoss().cuda()
        self.loss_bar = DLoss().cuda()
        self.loss_phrase = DLoss().cuda()

        # define lr
        self.lr_generator = self.config.learning_rate
        self.lr_discriminator = self.config.learning_rate
        self.lr_discriminator_feature = self.config.learning_rate
        self.lr_Zdiscriminator_bar = self.config.learning_rate
        self.lr_Zdiscriminator_phrase = self.config.learning_rate

        # define optimizer
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr_generator)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.opt_discriminator_feature = torch.optim.Adam(self.discriminator_feature.parameters(),
                                                          lr=self.lr_discriminator_feature)
        self.opt_Zdiscriminator_bar = torch.optim.Adam(self.z_discriminator_bar.parameters(),
                                                       lr=self.lr_Zdiscriminator_bar)
        self.opt_Zdiscriminator_phrase = torch.optim.Adam(self.z_discriminator_phrase.parameters(),
                                                          lr=self.lr_Zdiscriminator_phrase)

        # define optimize scheduler
        self.scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_generator, mode='min',
                                                                              factor=0.8, cooldown=6)
        self.scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_discriminator, mode='min',
                                                                                  factor=0.8, cooldown=6)
        self.scheduler_discriminator_feature = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_discriminator_feature,
                                                                                          mode='min', factor=0.8,
                                                                                          cooldown=6)
        self.scheduler_Zdiscriminator_bar = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_Zdiscriminator_bar,
                                                                                       mode='min', factor=0.8,
                                                                                       cooldown=6)
        self.scheduler_Zdiscriminator_phrase = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_Zdiscriminator_phrase,
                                                                                          mode='min', factor=0.8,
                                                                                          cooldown=6)

        # initialize counter
        self.iteration = 0
        self.epoch = 0

        self.manual_seed = random.randint(1, 10000)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        print("seed: ", self.manual_seed)

        # cuda setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.generator = nn.DataParallel(self.generator, device_ids=gpu_list)
        self.discriminator = nn.DataParallel(self.discriminator, device_ids=gpu_list)
        self.discriminator_feature = nn.DataParallel(self.discriminator_feature, device_ids=gpu_list)
        self.z_discriminator_bar = nn.DataParallel(self.z_discriminator_bar, device_ids=gpu_list)
        self.z_discriminator_phrase = nn.DataParallel(self.z_discriminator_phrase, device_ids=gpu_list)

        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()
        self.discriminator_feature = self.discriminator_feature.cuda()
        self.z_discriminator_bar = self.z_discriminator_bar.cuda()
        self.z_discriminator_phrase = self.z_discriminator_phrase.cuda()

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='BarGen')

        print('Number of generator parameters: {}'.format(sum([p.data.nelement() for p in self.generator.parameters()])))
        print('Number of discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.discriminator.parameters()])))
        print('Number of feature discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.discriminator_feature.parameters()])))
        print('Number of barZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_bar.parameters()])))
        print('Number of phraseZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_phrase.parameters()])))
        
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def set_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(filename="train_epoch.log")
    
        file_handler.setLevel(logging.WARNING)

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        
        return logger

    def make_batch(self, samples):
        note = np.concatenate([sample['note'] for sample in samples], axis=0)
        pre_note = np.concatenate([sample['pre_note'] for sample in samples], axis=0)
        pre_phrase = np.concatenate([sample['pre_phrase'] for sample in samples], axis=0)
        position = np.concatenate([sample['position'] for sample in samples], axis=0)

        return tuple([torch.tensor(note, dtype=torch.float), torch.tensor(pre_note, dtype=torch.float),
                      torch.tensor(pre_phrase, dtype=torch.float), torch.tensor(position, dtype=torch.long)])

    def free(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def frozen(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.opt_generator.load_state_dict(checkpoint['generator_optimizer'])

            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_discriminator.load_state_dict(checkpoint['disc_optimizer'])

            self.discriminator_feature.load_state_dict(checkpoint['discriminator_feature_state_dict'])
            self.opt_discriminator_feature.load_state_dict(checkpoint['disc_feature_optimizer'])

            self.z_discriminator_bar.load_state_dict(checkpoint['z_discriminator_bar_state_dict'])
            self.opt_Zdiscriminator_bar.load_state_dict(checkpoint['opt_Zdiscriminator_bar_optimizer'])

            self.z_discriminator_phrase.load_state_dict(checkpoint['z_discriminator_phrase_state_dict'])
            self.opt_Zdiscriminator_phrase.load_state_dict(checkpoint['opt_Zdiscriminator_phrase_optimizer'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, file_name, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer': self.opt_generator.state_dict(),

            'discriminator_state_dict': self.discriminator.state_dict(),
            'disc_optimizer': self.opt_discriminator.state_dict(),

            'discriminator_feature_state_dict': self.discriminator_feature.state_dict(),
            'disc_feature_optimizer': self.opt_discriminator_feature.state_dict(),

            'z_discriminator_bar_state_dict': self.z_discriminator_bar.state_dict(),
            'opt_Zdiscriminator_bar_optimizer': self.opt_Zdiscriminator_bar.state_dict(),

            'z_discriminator_phrase_state_dict': self.z_discriminator_phrase.state_dict(),
            'opt_Zdiscriminator_phrase_optimizer': self.opt_Zdiscriminator_phrase.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name,
                        os.path.join(self.config.root_path, self.config.checkpoint_dir, 'checkpoint.pth.tar'))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def record_image(self, image_sample, origin_image, outputs):
        self.summary_writer.add_image("train/sample 1", image_sample[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/sample 2", image_sample[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/sample 3", image_sample[2].reshape(1, 96, 60), self.epoch)

        image_sample = torch.gt(image_sample, 0.3).type('torch.cuda.FloatTensor')

        self.summary_writer.add_image("train/sample_binarization 1", image_sample[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/sample_binarization 2", image_sample[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/sample_binarization 3", image_sample[2].reshape(1, 96, 60), self.epoch)

        self.summary_writer.add_image("train/origin 1", origin_image[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/origin 2", origin_image[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("train/origin 3", origin_image[2].reshape(1, 96, 60), self.epoch)

        self.summary_writer.add_image("eval/generated 1", outputs[0].reshape(1, 96 * 4, 60), self.epoch)
        self.summary_writer.add_image("eval/generated 2", outputs[1].reshape(1, 96 * 4, 60), self.epoch)

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_epoch()

            if self.epoch > self.pretraining_step_size + 20:
                self.save_checkpoint(self.config.checkpoint_file, self.epoch)
            if self.epoch == self.pretraining_step_size:
                self.batch_size -= 2
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                             pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

    def train_epoch(self):
        if self.epoch > self.pretraining_step_size:
            self.train_count += 1
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}".format(self.epoch))

        image_sample = None
        origin_image = None
        Tensor = torch.cuda.FloatTensor

        avg_generator_loss = AverageMeter()
        avg_discriminator_loss = AverageMeter()
        avg_feature_discriminator_loss = AverageMeter()
        avg_barZ_disc_loss = AverageMeter()
        avg_phraseZ_disc_loss = AverageMeter()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            self.iteration += 1
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            note = Variable(note)
            pre_note = Variable(pre_note)
            pre_phrase = Variable(pre_phrase)
            position = Variable(position)

            origin_image = note
            valid_target = Variable(Tensor(note.size(0)).fill_(1.0), requires_grad=False)
            fake_target = Variable(Tensor(note.size(0)).fill_(0.0), requires_grad=False)

            if self.epoch <= self.pretraining_step_size:
                image_sample = self.train_pretrain(note, pre_note, pre_phrase, position, avg_generator_loss,
                                                   avg_barZ_disc_loss, avg_phraseZ_disc_loss,
                                                   fake_target, valid_target, curr_it)
            else:
                self.train_discriminator(note, pre_note, pre_phrase, position,
                                         avg_barZ_disc_loss, avg_phraseZ_disc_loss, avg_discriminator_loss,
                                         avg_feature_discriminator_loss, fake_target, valid_target)

                image_sample = self.train_add_gan(note, pre_note, pre_phrase, position,
                                                  avg_generator_loss, valid_target)

        tqdm_batch.close()
        
        self.scheduler_generator.step(avg_generator_loss.val)
        self.scheduler_Zdiscriminator_bar.step(avg_barZ_disc_loss.val)
        self.scheduler_Zdiscriminator_phrase.step(avg_phraseZ_disc_loss.val)
        if self.epoch > self.pretraining_step_size:
            self.scheduler_discriminator.step(avg_discriminator_loss.val)
            self.scheduler_discriminator_feature.step(avg_feature_discriminator_loss.val)

        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()
            self.discriminator_feature.eval()
            self.z_discriminator_bar.eval()
            self.z_discriminator_phrase.eval()

            outputs = []
            pre_phrase = torch.zeros(1, 1, 384, 60, dtype=torch.float32)
            pre_bar = torch.zeros(1, 1, 96, 60, dtype=torch.float32)
            phrase_idx = [330] + [i for i in range(10 - 2, -1, -1)]
            for idx in range(10):
                bar_set = []
                for _ in range(4):
                    pre_bar, _ = self.generator(torch.randn(1, 1152, dtype=torch.float32).cuda(), pre_bar.cuda(),
                                             pre_phrase, torch.from_numpy(np.array([phrase_idx[idx]])), False)
                    pre_bar = torch.gt(pre_bar, 0.3).type('torch.FloatTensor')  # 1, 1, 96, 96
                    bar_set.append(np.reshape(pre_bar.numpy(), [96, 60]))

                pre_phrase = np.concatenate(bar_set, axis=0)
                outputs.append(pre_phrase)
                pre_phrase = torch.from_numpy(np.reshape(pre_phrase, [1, 1, 96 * 4, 60])).float().cuda()

        self.record_image(image_sample[:3], origin_image[:3], outputs)

        self.logger.warning(
            'loss info - gen: {}, barZ disc: {},  phraseZ disc: {}, bar disc: {}, bar_seq disc: {}'.format(
                avg_generator_loss.val, avg_barZ_disc_loss.val, avg_phraseZ_disc_loss.val,
                avg_feature_discriminator_loss.val, avg_discriminator_loss.val)
        )
        self.logger.warning(
            'lr info - gen: {}, barZ disc: {},  phraseZ disc: {}, bar disc: {}, bar_seq disc: {}'.format(
                self.get_lr(self.opt_generator), self.get_lr(self.opt_Zdiscriminator_bar),
                self.get_lr(self.opt_Zdiscriminator_phrase), self.get_lr(self.opt_discriminator_feature),
                self.get_lr(self.opt_discriminator))
        )

    def train_discriminator(self,note, pre_note, pre_phrase, position,
                            avg_barZ_disc_loss, avg_phraseZ_disc_loss, avg_discriminator_loss,
                            avg_feature_discriminator_loss, fake_target, valid_target):
        self.free(self.discriminator)
        self.free(self.discriminator_feature)
        self.free(self.z_discriminator_bar)
        self.free(self.z_discriminator_phrase)

        self.frozen(self.generator)

        gen_note, z, pre_z, phrase_feature, gen_z = self.generator(note, pre_note, pre_phrase, position)

        #### Phrase Feature ###
        phrase_fake = (torch.randn(phrase_feature.size(0), phrase_feature.size(1)) * self.config.sigma).cuda()
        d_phrase_fake = self.z_discriminator_phrase(phrase_fake).view(-1)
        d_phrase_real = self.z_discriminator_phrase(phrase_feature).view(-1)
        phraseZ_dics_loss = self.loss_phrase(d_phrase_real, fake_target) + \
                            self.loss_phrase(d_phrase_fake, valid_target)

        #### Bar Feature ####
        bar_fake = (torch.randn(z.size(0), z.size(1)) * self.config.sigma).cuda()
        d_bar_fake = self.z_discriminator_bar(bar_fake).view(-1)
        d_bar_real = self.z_discriminator_bar(z).view(-1)
        barZ_dics_loss = self.loss_bar(d_bar_real, fake_target) + self.loss_bar(d_bar_fake, valid_target)

        #### Note ####
        out = torch.gt(gen_note, 0.3).type('torch.cuda.FloatTensor')
        fake_note = torch.cat((pre_note, out), dim=2)
        real_note = torch.cat((pre_note, note), dim=2)
        d_note_fake = self.discriminator(fake_note).view(-1)
        d_note_real = self.discriminator(real_note).view(-1)
        note_disc_loss = self.loss_disc(d_note_real, fake_target) + self.loss_disc(d_note_fake, valid_target)

        #### Note Feature ####
        d_feature_fake = self.discriminator_feature(gen_z).view(-1)
        d_feature_real = self.discriminator_feature(z).view(-1)
        feature_disc_loss = self.loss_disc(d_feature_real, fake_target) + \
                            self.loss_disc(d_feature_fake, valid_target)

        #######################
        phraseZ_dics_loss.backward()
        barZ_dics_loss.backward()
        note_disc_loss.backward()
        feature_disc_loss.backward()

        self.opt_Zdiscriminator_bar.step()
        self.opt_Zdiscriminator_phrase.step()
        self.opt_discriminator.step()
        self.opt_discriminator_feature.step()

        avg_barZ_disc_loss.update(barZ_dics_loss)
        avg_phraseZ_disc_loss.update(phraseZ_dics_loss)
        avg_discriminator_loss.update(note_disc_loss)
        avg_feature_discriminator_loss.update(feature_disc_loss)

        self.summary_writer.add_scalar("pre-train/Bar_Z_Discriminator_loss", avg_barZ_disc_loss.val, self.epoch)
        self.summary_writer.add_scalar("pre-train/Phrase_Z_discriminator_loss", avg_phraseZ_disc_loss.val, self.epoch)
        self.summary_writer.add_scalar("train/Bar_Discriminator_loss", avg_discriminator_loss.val, self.epoch)
        self.summary_writer.add_scalar("train/Bar_Feature_discriminator_loss", avg_feature_discriminator_loss.val,
                                       self.epoch)

    def train_pretrain(self, note, pre_note, pre_phrase, position, avg_generator_loss,
                       avg_barZ_disc_loss, avg_phraseZ_disc_loss, fake_target, valid_target, curr_it):
        self.generator.train()
        self.z_discriminator_bar.train()
        self.z_discriminator_phrase.train()

        self.discriminator.eval()
        self.discriminator_feature.eval()

        self.generator.zero_grad()
        self.z_discriminator_bar.zero_grad()
        self.z_discriminator_phrase.zero_grad()

        gen_note, z, pre_z, phrase_feature, gen_z = self.generator(note, pre_note, pre_phrase, position)

        #### Phrase Feature ###
        phrase_fake = (torch.randn(phrase_feature.size(0), phrase_feature.size(1)) * self.config.sigma).cuda()
        d_phrase_fake = self.z_discriminator_phrase(phrase_fake).view(-1)
        d_phrase_real = self.z_discriminator_phrase(phrase_feature).view(-1)
        phraseZ_dics_loss = self.loss_phrase(d_phrase_real, fake_target) + \
                            self.loss_phrase(d_phrase_fake, valid_target)

        #### Bar Feature ####
        bar_fake = (torch.randn(z.size(0), z.size(1)) * self.config.sigma).cuda()
        d_bar_fake = self.z_discriminator_bar(bar_fake).view(-1)
        d_bar_real = self.z_discriminator_bar(z).view(-1)
        barZ_dics_loss = self.loss_bar(d_bar_real, fake_target) + self.loss_bar(d_bar_fake, valid_target)

        #######################
        phraseZ_dics_loss.backward()
        barZ_dics_loss.backward()

        self.opt_Zdiscriminator_bar.step()
        self.opt_Zdiscriminator_phrase.step()

        avg_barZ_disc_loss.update(barZ_dics_loss)
        avg_phraseZ_disc_loss.update(phraseZ_dics_loss)

        #################### Generator ####################
        self.free(self.generator)

        self.frozen(self.discriminator)
        self.frozen(self.discriminator_feature)
        self.frozen(self.z_discriminator_bar)
        self.frozen(self.z_discriminator_phrase)

        gen_note, z, pre_z, phrase_feature, _ = self.generator(note, pre_note, pre_phrase, position)

        #### Generator Loss ####
        loss = self.loss_generator(gen_note, note, True)

        loss.backward()

        self.opt_generator.step()

        avg_generator_loss.update(loss)
        self.summary_writer.add_scalar("pre-train/Generator_loss", avg_generator_loss.val, self.epoch)
        self.summary_writer.add_scalar("pre-train/Bar_Z_Discriminator_loss", avg_barZ_disc_loss.val, self.epoch)
        self.summary_writer.add_scalar("pre-train/Phrase_Z_discriminator_loss", avg_phraseZ_disc_loss.val, self.epoch)

        return gen_note[:3]

    def train_add_gan(self, note, pre_note, pre_phrase, position, avg_generator_loss, valid_target):
        self.generator.train()
        self.z_discriminator_bar.train()
        self.z_discriminator_phrase.train()

        self.discriminator.eval()
        self.discriminator_feature.eval()

        self.generator.zero_grad()
        self.z_discriminator_bar.zero_grad()
        self.z_discriminator_phrase.zero_grad()

        #################### Generator ####################
        self.free(self.generator)

        self.frozen(self.z_discriminator_bar)
        self.frozen(self.z_discriminator_phrase)
        self.frozen(self.discriminator)
        self.frozen(self.discriminator_feature)

        gen_note, z, pre_z, phrase_feature, _ = self.generator(note, pre_note, pre_phrase, position)

        #### Phrase Encoder Loss ###
        d_phrase_real = self.z_discriminator_phrase(phrase_feature).view(-1)
        loss = self.loss_phrase(d_phrase_real, valid_target)

        #### Bar Encoder Loss ####
        d_bar_real1 = self.z_discriminator_bar(z).view(-1)
        d_bar_real2 = self.z_discriminator_bar(pre_z).view(-1)
        loss += self.loss_bar(d_bar_real1, valid_target) + self.loss_bar(d_bar_real2, valid_target)

        #### Generator Loss ####
        loss += self.loss_generator(gen_note, note, False)

        #### Gan Loss ####
        noise = torch.normal(0, 1.5, size=(note.size(0), 1152))
        gen_note, gen_z = self.generator(noise, pre_note, pre_phrase, position, False)

        out = torch.gt(gen_note, 0.3).type('torch.cuda.FloatTensor')
        d_note_fake = self.discriminator(torch.cat((pre_note, out), dim=2)).view(-1)
        loss += self.loss_disc(d_note_fake, valid_target) * 0.05

        d_feature_fake = self.discriminator_feature(gen_z).view(-1)
        loss += self.loss_disc(d_feature_fake, valid_target) * 0.05

        loss.backward()

        self.opt_generator.step()

        avg_generator_loss.update(loss)

        self.summary_writer.add_scalar("train/Generator_loss", avg_generator_loss.val, self.epoch)

        return gen_note[:3]
