import os
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
from graph.phrase_encoder import PhraseModel
from graph.bar_discriminator import BarDiscriminator

from graph.loss.bar_loss import Loss, PhraseLoss, DLoss
from data.bar_dataset import NoteDataset, TestDataset

from metrics import AverageMeter
from tensorboardX import SummaryWriter

cudnn.benchmark = True


class BarGen(object):
    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("BarGen")

        self.batch_size = self.config.batch_size

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        self.testset = TestDataset(self.config.root_path, self.config)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        # define models ( generator and discriminator)
        self.generator = Model()
        self.discriminator = BarDiscriminator()
        self.phrase = PhraseModel([])

        # define loss
        self.loss_gen = Loss().cuda()
        self.loss_disc = DLoss().cuda()
        self.loss_phrase = PhraseLoss().cuda()

        # define lr
        self.lr_gen = self.config.learning_rate
        self.lr_phrase = self.config.learning_rate

        self.GAN_lr_gen = self.config.learning_rate / 2
        self.GAN_lr_disc = self.config.learning_rate / 2
        self.GAN_lr_phrase = self.config.learning_rate / 2

        # define optimizer
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen)
        self.opt_phrase = torch.optim.Adam(self.phrase.parameters(), lr=self.lr_phrase)

        self.GAN_opt_gen = torch.optim.Adam(self.discriminator.parameters(), lr=self.GAN_lr_gen)
        self.GAN_opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.GAN_lr_disc)
        self.GAN_opt_phrase = torch.optim.Adam(self.discriminator.parameters(), lr=self.GAN_lr_phrase)

        # define optimize scheduler
        self.scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_gen, mode='min', factor=0.8,
                                                                        cooldown=5)
        self.scheduler_phrase = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_phrase, mode='min', factor=0.8,
                                                                           cooldown=5)

        self.scheduler_GAN_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(self.GAN_opt_gen, mode='min', factor=0.8,
                                                                            cooldown=5)
        self.scheduler_GAN_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(self.GAN_opt_disc, mode='min', factor=0.8,
                                                                             cooldown=5)
        self.scheduler_GAN_phrase = torch.optim.lr_scheduler.ReduceLROnPlateau(self.GAN_opt_phrase, mode='min',
                                                                               factor=0.8, cooldown=5)

        # initialize counter
        self.vae_iteration = 0
        self.gan_iteration = 0
        self.current_epoch = 0
        self.vae_best = 9999999999.
        self.gan_best = 9999999999.

        self.manual_seed = random.randint(1, 10000)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        print("seed: ", self.manual_seed)

        # cuda setting
        if len(self.config.gpu_device) > 1:
            self.generator = nn.DataParallel(self.generator, device_ids=list(range(self.config.gpu_cnt)))
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=list(range(self.config.gpu_cnt)))
            self.phrase = nn.DataParallel(self.phrase, device_ids=list(range(self.config.gpu_cnt)))

        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()
        self.phrase = self.phrase.cuda()

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='MC_VAE')

        print('Number of generator parameters: {}'.format(sum([p.data.nelement() for p in self.generator.parameters()])))
        print('Number of discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.discriminator.parameters()])))
        print('Number of phrase parameters: {}'.format(sum([p.data.nelement() for p in self.phrase.parameters()])))

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
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.opt_gen.load_state_dict(checkpoint['gen_optimizer'])
            self.GAN_opt_gen.load_state_dict(checkpoint['GAN_gen_optimizer'])

            self.phrase.load_state_dict(checkpoint['phrase_state_dict'])
            self.opt_phrase.load_state_dict(checkpoint['phrase_optimizer'])
            self.GAN_opt_phrase.load_state_dict(checkpoint['GAN_phrase_optimizer'])

            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.GAN_opt_disc.load_state_dict(checkpoint['GAN_disc_optimizer'])

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name, epoch, is_best=False):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch))
        # file_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)

        state = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'gen_optimizer': self.opt_gen,
            'GAN_gen_optimizer': self.GAN_opt_gen,

            'phrase_state_dict': self.phrase,
            'phrase_optimizer': self.opt_phrase,
            'GAN_phrase_optimizer': self.GAN_opt_phrase,

            'discriminator_state_dict': self.discriminator,
            'GAN_disc_optimizer': self.GAN_opt_disc,
        }

        torch.save(state, tmp_name)
        if is_best:
            shutil.copyfile(tmp_name,
                            os.path.join(self.config.root_path, self.config.checkpoint_dir, 'model_best.pth.tar'))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.config.epoch):
            self.current_epoch += 1
            is_best, loss = self.train_vae()
            if epoch > 300:
                self.save_checkpoint(self.config.checkpoint_file, epoch, is_best)

            lr = 0.
            for param_group in self.opt_gen.param_groups:
                lr = param_group['lr']

            print('{}epoch loss: {}, lr: {}'.format(self.current_epoch, loss, lr))

        for epoch in range(self.config.epoch):
            self.current_epoch += 1

            train_gen = True
            if self.test_disc() < 0.65:
                train_gen = False

            is_best, loss = self.train_gan(train_gen)

            if train_gen:
                self.save_checkpoint(self.config.checkpoint_file, epoch, is_best)

            lr = 0.
            for param_group in self.opt_gen.param_groups:
                lr = param_group['lr']

            print('{}epoch loss: {}, lr: {}'.format(self.current_epoch, loss, lr))

    def train_vae(self):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        self.generator.train()
        self.phrase.train()

        gen_avg_loss = AverageMeter()
        phrase_avg_loss = AverageMeter()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            ####################
            self.generator.zero_grad()
            self.phrase.zero_grad()

            #################### Generator ####################
            self.free(self.generator)
            self.frozen(self.phrase)

            phrase_feature, _, _ = self.phrase(pre_phrase, position)
            gen_note, mean, var, pre_mean, pre_var, z, z_gen = self.generator(note, pre_note, phrase_feature)

            gen_loss = self.loss_gen(gen_note, note, mean, var, pre_mean, pre_var, z, z_gen)
            gen_loss.backward(retain_graph=True)
            self.opt_gen.step()

            #################### Phrase Encoder ####################
            self.free(self.phrase)
            self.frozen(self.generator)

            phrase_feature, mean, var = self.phrase(pre_phrase, position)
            gen_note, _, _, _, _, _, _ = self.generator(note, pre_note, phrase_feature)

            phrase_loss = self.loss_phrase(gen_note, note, mean, var)
            phrase_loss.backward(retain_graph=True)
            self.opt_phrase.step()

            ####################
            gen_avg_loss.update(gen_loss.item())
            phrase_avg_loss.update(phrase_loss.item())

            self.vae_iteration += 1

            self.summary_writer.add_scalar("vae/Generator_loss", gen_avg_loss.val, self.vae_iteration)
            self.summary_writer.add_scalar("vae/Phrase_loss", phrase_avg_loss.val, self.vae_iteration)

        tqdm_batch.close()
        self.scheduler_gen.step(gen_avg_loss.val)
        self.scheduler_phrase.step(phrase_avg_loss.val)

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Discriminator loss: "
                         + " - Generator Loss-: " + str(gen_avg_loss.val))

        if gen_avg_loss.val < self.vae_best:
            self.vae_best = gen_avg_loss.val
            return True, gen_avg_loss.val
        else:
            return False, gen_avg_loss.val

    def train_gan(self, train_gen=True):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        self.generator.train()
        self.discriminator.train()
        self.phrase.train()

        gen_avg_loss = AverageMeter()
        disc_avg_loss = AverageMeter()
        phrase_avg_loss = AverageMeter()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            gen_loss, disc_loss = None, None

            ####################
            if train_gen:
                self.generator.zero_grad()
                self.phrase.zero_grad()

                #################### Generator ####################
                self.free(self.generator)
                self.frozen(self.phrase)

                phrase_feature, _, _ = self.phrase(pre_phrase, position)
                gen_note, mean, var, pre_mean, pre_var, z, z_gen = self.generator(note, pre_note, phrase_feature)

                gen_loss = self.loss_gen(gen_note, note, mean, var, pre_mean, pre_var, z, z_gen)
                gen_loss.backward(retain_graph=True)
                self.GAN_opt_gen.step()

            else:
                self.discriminator.zero_grad()
                self.phrase.zero_grad()

                #################### Generator ####################
                self.free(self.discriminator)
                self.frozen(self.phrase)

                phrase_feature, _, _ = self.phrase(pre_phrase, position)
                gen_note, mean, var, pre_mean, pre_var, z, z_gen = self.generator(note, pre_note, phrase_feature)

                disc_loss = self.loss_gen(gen_note, note, mean, var, pre_mean, pre_var, z, z_gen)
                disc_loss.backward(retain_graph=True)
                self.GAN_opt_disc.step()


            #################### Phrase Encoder ####################
            self.free(self.phrase)
            self.frozen(self.generator)
            self.frozen(self.discriminator)

            phrase_feature, mean, var = self.phrase(pre_phrase, position)
            gen_note, _, _, _, _, _, _ = self.generator(note, pre_note, phrase_feature)

            phrase_loss = self.loss_phrase(gen_note, note, mean, var)
            phrase_loss.backward(retain_graph=True)
            self.GAN_opt_phrase.step()

            ####################
            self.gan_iteration += 1
            phrase_avg_loss.update(phrase_loss.item())
            self.summary_writer.add_scalar("gan/Phrase_loss", phrase_avg_loss.val, self.gan_iteration)

            if train_gen:
                gen_avg_loss.update(gen_loss.item())
                self.summary_writer.add_scalar("gan/Generator_loss", gen_avg_loss.val, self.gan_iteration)
            else:
                disc_avg_loss.update(disc_loss.item())
                self.summary_writer.add_scalar("gan/Generator_loss", disc_avg_loss.val, self.gan_iteration)

        tqdm_batch.close()
        if train_gen:
            self.scheduler_GAN_gen.step(gen_avg_loss.val)
        else:
            self.scheduler_GAN_disc.step(disc_avg_loss.val)
        self.scheduler_GAN_phrase.step(phrase_avg_loss.val)

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Discriminator loss: "
                         + " - Generator Loss-: " + str(gen_avg_loss.val))

        if gen_avg_loss.val < self.gan_best:
            self.gan_best = gen_avg_loss.val
            return True, gen_avg_loss.val
        else:
            return False, gen_avg_loss.val

    def test_disc(self):
        tqdm_batch = tqdm(self.testloader, total=self.dataset.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        self.generator.eval()
        self.discriminator.eval()
        self.phrase.eval()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            phrase_feature, _, _ = self.phrase(pre_phrase, position)
            gen_note, _, _, _, _, _ = self.generator(note, pre_note, phrase_feature)
            gen_note = torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor')

            fake_note = torch.cat((pre_note, gen_note), dim=2)
            fake_target = torch.zeros(fake_note.size[0])

            real_note = torch.cat((pre_note, note), dim=2)
            real_target = torch.ones(real_note.size[0])

            note = torch.cat((fake_note, real_note), dim=0)
            target = torch.cat((fake_target, real_target), dim=0)

            logits = self.discriminator(note)
            output = logits > 0.5

            return (target == output).sum().float() / target.size[0]



