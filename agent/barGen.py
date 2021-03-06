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
from graph.model import Model
from graph.bar_discriminator import BarDiscriminator
from graph.z_discriminator import PhraseZDiscriminator, BarZDiscriminator

from graph.loss.bar_loss import Loss, DLoss
from data.bar_dataset import NoteDataset

from metrics import AverageMeter
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class BarGen(object):
    def __init__(self, config):
        self.config = config

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size
        
        self.logger = self.set_logger()

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        # define models ( generator and discriminator)
        self.generator = Model()
        self.discriminator = BarDiscriminator()
        self.z_discriminator_phrase = PhraseZDiscriminator()
        self.z_discriminator_bar = BarZDiscriminator()

        # define loss
        self.loss_gen = Loss().cuda()
        self.loss_disc = DLoss().cuda()

        # define lr
        self.lr_gen1 = self.config.learning_rate
        self.lr_gen2 = self.config.learning_rate
        self.lr_discriminator = self.config.learning_rate
        self.lr_Zdiscriminator_bar = self.config.learning_rate
        self.lr_Zdiscriminator_phrase = self.config.learning_rate

        # define optimizer
        self.opt_gen1 = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen1)
        self.opt_gen2 = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen2)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.opt_Zdiscriminator_bar = torch.optim.Adam(self.z_discriminator_bar.parameters(),
                                                       lr=self.lr_Zdiscriminator_bar)
        self.opt_Zdiscriminator_phrase = torch.optim.Adam(self.z_discriminator_phrase.parameters(),
                                                          lr=self.lr_Zdiscriminator_phrase)

        # define optimize scheduler
        self.scheduler_gen1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_gen1, mode='min', factor=0.8,
                                                                        cooldown=6)
        self.scheduler_gen2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_gen2, mode='min', factor=0.8,
                                                                        cooldown=6)
        self.scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_discriminator, mode='min',
                                                                                  factor=0.8, cooldown=6)
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
        self.z_discriminator_bar = nn.DataParallel(self.z_discriminator_bar, device_ids=gpu_list)
        self.z_discriminator_phrase = nn.DataParallel(self.z_discriminator_phrase, device_ids=gpu_list)

        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()
        self.z_discriminator_bar = self.z_discriminator_bar.cuda()
        self.z_discriminator_phrase = self.z_discriminator_phrase.cuda()

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='BarGen')

        print('Number of generator parameters: {}'.format(sum([p.data.nelement() for p in self.generator.parameters()])))
        print('Number of discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.discriminator.parameters()])))
        print('Number of barZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_bar.parameters()])))
        print('Number of phraseZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_phrase.parameters()])))
        
    def set_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(filename="train_epoch.log")
    
        file_handler.setLevel(logging.DEBUG)

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
            self.opt_gen1.load_state_dict(checkpoint['gen_optimizer1'])
            self.opt_gen2.load_state_dict(checkpoint['gen_optimizer2'])

            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_discriminator.load_state_dict(checkpoint['disc_optimizer'])

            self.z_discriminator_bar.load_state_dict(checkpoint['z_discriminator_bar_state_dict'])
            self.opt_Zdiscriminator_bar.load_state_dict(checkpoint['opt_Zdiscriminator_bar_optimizer'])

            self.z_discriminator_phrase.load_state_dict(checkpoint['z_discriminator_phrase_state_dict'])
            self.opt_Zdiscriminator_phrase.load_state_dict(checkpoint['opt_Zdiscriminator_phrase_optimizer'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, file_name, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch))
        # file_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)

        state = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'gen_optimizer1': self.opt_gen1.state_dict(),
            'gen_optimizer2': self.opt_gen2.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'disc_optimizer': self.opt_discriminator.state_dict(),
            'z_discriminator_bar_state_dict': self.z_discriminator_bar.state_dict(),
            'opt_Zdiscriminator_bar_optimizer': self.opt_Zdiscriminator_bar.state_dict(),
            'z_discriminator_phrase_state_dict': self.z_discriminator_phrase.state_dict(),
            'opt_Zdiscriminator_phrase_optimizer': self.opt_Zdiscriminator_phrase.state_dict(),
            'lr_gen': self.config.learning_rate,
            'lr_discriminator': self.config.learning_rate,
            'lr_Zdiscriminator_bar': self.config.learning_rate,
            'lr_Zdiscriminator_phrase': self.config.learning_rate,
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name,
                        os.path.join(self.config.root_path, self.config.checkpoint_dir, 'checkpoint.pth.tar'))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.config.epoch):
            self.epoch += 1
            self.train_epoch()

            if self.epoch > self.pretraining_step_size + 50:
                self.save_checkpoint(self.config.checkpoint_file, self.epoch)

    def train_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}".format(self.epoch))

        self.generator.train()
        self.discriminator.train()
        self.z_discriminator_bar.train()
        self.z_discriminator_phrase.train()

        image_sample = None
        Tensor = torch.cuda.FloatTensor

        avg_gen_loss = AverageMeter()
        avg_disc_loss = AverageMeter()
        avg_barZ_disc_loss = AverageMeter()
        avg_phraseZ_disc_loss = AverageMeter()
        div_flag = random.randrange(2, 5)
        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            note = Variable(note)
            pre_note = Variable(pre_note)
            pre_phrase = Variable(pre_phrase)
            position = Variable(position)

            valid_target = Variable(Tensor(note.size(0)).fill_(1.0), requires_grad=False)
            fake_target = Variable(Tensor(note.size(0)).fill_(0.0), requires_grad=False)
            fake_target_double = Variable(Tensor(note.size(0) * 2).fill_(0.0), requires_grad=False)

            self.iteration += 1

            ####################
            self.generator.zero_grad()
            self.discriminator.zero_grad()
            self.z_discriminator_bar.zero_grad()
            self.z_discriminator_phrase.zero_grad()

            if (curr_it + self.epoch) % div_flag == 1 and self.epoch > self.pretraining_step_size:
                #################### Discriminator ####################
                self.free(self.discriminator)
                self.free(self.z_discriminator_bar)
                self.free(self.z_discriminator_phrase)

                self.frozen(self.generator)

                gen_note, z, pre_z, phrase_feature = self.generator(note, pre_note, pre_phrase, position)

                #### Phrase Feature ###
                phrase_fake = torch.randn(phrase_feature.size(0), phrase_feature.size(1)).cuda()
                d_phrase_fake = self.z_discriminator_phrase(phrase_fake).view(-1)
                d_phrase_real = self.z_discriminator_phrase(phrase_feature).view(-1)
                phraseZ_dics_loss = self.loss_disc(d_phrase_real, valid_target) + self.loss_disc(d_phrase_fake, fake_target)

                #### Bar Feature ####
                bar_fake = torch.randn(z.size(0) * 2, z.size(1)).cuda()
                d_bar_fake = self.z_discriminator_bar(bar_fake).view(-1)
                d_bar_real1 = self.z_discriminator_bar(z).view(-1)
                d_bar_real2 = self.z_discriminator_bar(pre_z).view(-1)
                barZ_dics_loss = self.loss_disc(d_bar_real1, valid_target) + self.loss_disc(d_bar_real2, valid_target) +\
                                 self.loss_disc(d_bar_fake, fake_target_double)

                #### Generated Bar ####
                fake_note = torch.gt(gen_note, 0.3).type('torch.cuda.FloatTensor')
                fake_note = torch.cat((pre_note, fake_note), dim=2)
                d_fake = self.discriminator(fake_note).view(-1)

                real_note = torch.cat((pre_note, note), dim=2)
                d_real = self.discriminator(real_note).view(-1)

                disc_loss = self.loss_disc(d_fake, fake_target) + self.loss_disc(d_real, valid_target)

                #######################
                disc_loss.backward()
                phraseZ_dics_loss.backward()
                barZ_dics_loss.backward()

                self.opt_discriminator.step()
                self.opt_Zdiscriminator_bar.step()
                self.opt_Zdiscriminator_phrase.step()

                avg_disc_loss.update(disc_loss)
                avg_barZ_disc_loss.update(barZ_dics_loss)
                avg_phraseZ_disc_loss.update(phraseZ_dics_loss)

            #################### Generator ####################
            self.free(self.generator)

            self.frozen(self.discriminator)
            self.frozen(self.z_discriminator_bar)
            self.frozen(self.z_discriminator_phrase)

            gen_note, z, pre_z, phrase_feature = self.generator(note, pre_note, pre_phrase, position)
            image_sample = gen_note
            origin_image = note

            #### add GAN Loss ###
            if self.epoch > self.pretraining_step_size:
                gen_loss = self.loss_gen(gen_note, note, False)
                
                gen_loss = self.loss_disc(self.z_discriminator_phrase(phrase_feature).view(-1), valid_target)
                gen_loss += self.loss_disc(self.z_discriminator_bar(z).view(-1), valid_target) + \
                            self.loss_disc(self.z_discriminator_bar(pre_z).view(-1), valid_target)

                fake_note = torch.gt(gen_note, 0.3).type('torch.cuda.FloatTensor')
                fake_note = torch.cat((pre_note, fake_note), dim=2)
                d_fake = self.discriminator(fake_note).view(-1)

                gen_loss += self.loss_disc(d_fake, valid_target)

                gen_loss.backward()
                self.opt_gen2.step()

            else:
                gen_loss = self.loss_gen(gen_note, note, True)
                
                gen_loss.backward()
                self.opt_gen1.step()

            avg_gen_loss.update(gen_loss.item())

            if self.epoch > self.pretraining_step_size:
                self.summary_writer.add_scalar("train/Generator_loss", avg_gen_loss.val, self.iteration)
                self.summary_writer.add_scalar("train/Discriminator_loss", avg_disc_loss.val, self.iteration)
                self.summary_writer.add_scalar("train/Bar_Z_Discriminator_loss", avg_barZ_disc_loss.val, self.iteration)
                self.summary_writer.add_scalar("train/Phrase_Z_discriminator_loss", avg_phraseZ_disc_loss.val, self.iteration)
            else:
                self.summary_writer.add_scalar("pre_train/Generator_loss", avg_gen_loss.val, self.iteration)

        tqdm_batch.close()

        self.summary_writer.add_image("generated/sample 1_1", image_sample[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/sample 1_2", image_sample[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/sample 1_3", image_sample[2].reshape(1, 96, 60), self.epoch)

        image_sample = torch.gt(image_sample, 0.3).type('torch.cuda.FloatTensor')

        self.summary_writer.add_image("generated/sample 2_1", image_sample[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/sample 2_2", image_sample[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/sample 2_3", image_sample[2].reshape(1, 96, 60), self.epoch)

        self.summary_writer.add_image("generated/origin 2_1", origin_image[0].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/origin 2_2", origin_image[1].reshape(1, 96, 60), self.epoch)
        self.summary_writer.add_image("generated/origin 2_3", origin_image[2].reshape(1, 96, 60), self.epoch)

        if self.epoch > self.pretraining_step_size:
            self.scheduler_gen1.step(avg_gen_loss.val)
        else:
            self.scheduler_gen2.step(avg_gen_loss.val)
        self.scheduler_discriminator.step(avg_disc_loss.val)
        self.scheduler_Zdiscriminator_bar.step(avg_barZ_disc_loss.val)
        self.scheduler_Zdiscriminator_phrase.step(avg_phraseZ_disc_loss.val)

        self.logger.debug('pre_train lr: {},  generator lr: {},  bar disc lr: {},  barZ disc lr: {},  phraseZ disc lr: {}'.format(self.lr_gen1, self.lr_gen2, self.lr_discriminator,
                                                                                                                                  self.lr_Zdiscriminator_bar, self.lr_Zdiscriminator_phrase))