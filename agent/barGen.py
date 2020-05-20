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

        self.batch_size = self.config.batch_size

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
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
        self.lr_gen = self.config.learning_rate
        self.lr_discriminator = self.config.learning_rate
        self.lr_Zdiscriminator_bar = self.config.learning_rate
        self.lr_Zdiscriminator_phrase = self.config.learning_rate

        # define optimizer
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.opt_Zdiscriminator_bar = torch.optim.Adam(self.z_discriminator_bar.parameters(),
                                                       lr=self.lr_Zdiscriminator_bar)
        self.opt_Zdiscriminator_phrase = torch.optim.Adam(self.z_discriminator_phrase.parameters(),
                                                          lr=self.lr_Zdiscriminator_phrase)

        # define optimize scheduler
        self.scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_gen, mode='min', factor=0.8,
                                                                        cooldown=5)
        self.scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_discriminator, mode='min',
                                                                                  factor=0.8, cooldown=5)
        self.scheduler_Zdiscriminator_bar = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_Zdiscriminator_bar,
                                                                                       mode='min', factor=0.8,
                                                                                       cooldown=5)
        self.scheduler_Zdiscriminator_phrase = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_Zdiscriminator_phrase,
                                                                                          mode='min', factor=0.8,
                                                                                          cooldown=5)

        # initialize counter
        self.vae_iteration = 0
        self.gan_iteration = 0
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

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='BarGen')

        print('Number of generator parameters: {}'.format(sum([p.data.nelement() for p in self.generator.parameters()])))
        print('Number of discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.discriminator.parameters()])))
        print('Number of barZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_bar.parameters()])))
        print('Number of phraseZ discriminator parameters: {}'.format(sum([p.data.nelement() for p in self.z_discriminator_phrase.parameters()])))

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
            self.opt_gen.load_state_dict(checkpoint['gen_optimizer'])

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
            'gen_optimizer': self.opt_gen.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'disc_optimizer': self.opt_discriminator.state_dict(),
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

    def train(self):
        for epoch in range(self.config.epoch):
            self.epoch += 1
            self.train()
            self.save_checkpoint(self.config.checkpoint_file, epoch)

    def train(self):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}".format(self.epoch))

        self.generator.train()
        self.discriminator.train()
        self.z_discriminator_bar.train()
        self.z_discriminator_phrase.train()

        avg_gen_loss = AverageMeter()
        avg_disc_loss = AverageMeter()
        avg_barZ_disc_loss = AverageMeter()
        avg_phraseZ_disc_loss = AverageMeter()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            note = note.cuda(async=self.config.async_loading)
            pre_note = pre_note.cuda(async=self.config.async_loading)
            pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
            position = position.cuda(async=self.config.async_loading)

            ####################
            self.generator.zero_grad()
            self.discriminator.zero_grad()
            self.z_discriminator_bar.zero_grad()
            self.z_discriminator_phrase.zero_grad()

            #################### Discriminator ####################
            self.free(self.discriminator)
            self.free(self.z_discriminator_bar)
            self.free(self.z_discriminator_phrase)

            self.frozen(self.generator)

            gen_note, z, pre_z, phrase_feature = self.generator(Variable(note), Variable(pre_note),
                                                                Variable(pre_phrase), Variable(position))

            #### Phrase Feature ###
            phrase_fake = torch.randn(phrase_feature.size(0), phrase_feature.size(1)).cuda()
            d_phrase_fake = self.z_discriminator_phrase(phrase_fake)
            d_phrase_real = self.z_discriminator_phrase(phrase_feature)
            phraseZ_dics_loss = self.loss_disc(d_phrase_real, 1) + self.loss_disc(d_phrase_fake)

            #### Bar Feature ####
            bar_fake = torch.randn(z.size(0) * 2, z.size(1)).cuda()
            d_bar_fake = self.z_discriminator_bar(bar_fake)
            d_bar_real1 = self.z_discriminator_bar(z)
            d_bar_real2 = self.z_discriminator_bar(pre_z)
            barZ_dics_loss = self.loss_disc(d_bar_real1, 1) + self.loss_disc(d_bar_real2, 1) + self.loss_disc(d_bar_fake)

            #### Generated Bar ####
            fake_note = torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor')
            fake_note = torch.cat((pre_note, fake_note), dim=2)
            d_fake = self.discriminator(fake_note).view(-1)

            real_note = torch.cat((pre_note, note), dim=2)
            d_real = self.discriminator(real_note).view(-1)

            disc_loss = self.loss_disc(d_fake) + self.loss_disc(d_real, 1)

            #######################
            disc_loss.backward(retain_graph=True)
            phraseZ_dics_loss.backward(retain_graph=True)
            barZ_dics_loss.backward(retain_graph=True)

            self.opt_discriminator.step()
            self.opt_Zdiscriminator_bar.step()
            self.opt_Zdiscriminator_phrase.step()

            #################### Generator ####################
            self.free(self.generator)

            self.frozen(self.discriminator)
            self.frozen(self.z_discriminator_bar)
            self.frozen(self.z_discriminator_phrase)

            gen_note, z, pre_z, phrase_feature = self.generator(Variable(note), Variable(pre_note),
                                                                Variable(pre_phrase), Variable(position))

            #### Discriminator Loss ###
            gan_loss = self.loss_disc(self.z_discriminator_phrase(phrase_feature), 1)

            gan_loss += self.loss_disc(self.z_discriminator_bar(z), 1) + \
                        self.loss_disc(self.z_discriminator_bar(pre_z), 1)

            fake_note = torch.gt(gen_note, 0.35).type('torch.cuda.FloatTensor')
            fake_note = torch.cat((pre_note, fake_note), dim=2)
            d_fake = self.discriminator(fake_note).view(-1)

            gan_loss += self.loss_disc(d_fake, 1)

            gen_loss = self.loss_gen(gen_note, note, gan_loss)

            gen_loss.backward(retain_graph=True)

            self.opt_gen.step()

            ####################
            avg_gen_loss.update(gen_loss.item())
            avg_disc_loss.update(disc_loss)
            avg_barZ_disc_loss.update(barZ_dics_loss)
            avg_phraseZ_disc_loss.update(phraseZ_dics_loss)

            self.vae_iteration += 1

        tqdm_batch.close()

        self.summary_writer.add_scalar("epoch/Generator_loss", avg_gen_loss.val, self.epoch)
        self.summary_writer.add_scalar("epoch/Discriminator_loss", avg_disc_loss.val, self.epoch)
        self.summary_writer.add_scalar("epoch/Bar_Z_Discriminator_loss", avg_barZ_disc_loss.val, self.epoch)
        self.summary_writer.add_scalar("epoch/Phrase_Z_discriminator_loss", avg_phraseZ_disc_loss.val, self.epoch)

        self.scheduler_gen.step(avg_gen_loss.val)
        self.scheduler_discriminator.step(avg_disc_loss)
        self.scheduler_Zdiscriminator_bar.step(avg_barZ_disc_loss)
        self.scheduler_Zdiscriminator_phrase.step(avg_phraseZ_disc_loss)

