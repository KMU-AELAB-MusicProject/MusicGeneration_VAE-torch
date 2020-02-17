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
from graphs.models.bar_v1.model import Model
from graphs.models.bar_v1.phrase_model import PhraseModel
from graphs.models.bar_discriminator import BarDiscriminator
from graphs.losses.bar_loss import Loss, DLoss, PhraseLoss
from datasets.bar_dataset import NoteDataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate

cudnn.benchmark = True


class MCVAE(object):
    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("MC_VAE")

        self.batch_size = self.config.batch_size

        # define models ( generator and discriminator)
        self.model = Model()
        self.phrase_model = PhraseModel()
        self.discriminator = BarDiscriminator()

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=3,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        # define loss
        self.loss = Loss()
        self.phrase_loss = PhraseLoss()
        self.lossD = DLoss()

        # define optimizers for both generator and discriminator
        self.lr = self.config.learning_rate
        self.lr_phrase = self.config.learning_rate
        self.lrD = self.config.learning_rate
        self.optimVAE = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optim_phrase = torch.optim.Adam(self.phrase_model.parameters(), lr=self.lr_phrase)
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lrD)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_error = 9999999999.

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        self.manual_seed = random.randint(1, 10000)

        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        if self.cuda:
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device[0])
            self.device = torch.device("cuda")

        else:
            self.logger.info("Program will run on *****CPU***** ")
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.phrase_model = self.phrase_model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.loss = self.loss.to(self.device)
        self.phrase_loss = self.phrase_loss.to(self.device)
        self.lossD = self.lossD.to(self.device)

        if len(self.config.gpu_device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_device)
            self.phrase_model = nn.DataParallel(self.phrase_model, device_ids=self.config.gpu_device)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.config.gpu_device)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='MC_VAE')

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

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimVAE.load_state_dict(checkpoint['model_optimizer'])
            self.phrase_model.load_state_dict(checkpoint['phrase_model_state_dict'])
            self.optim_phrase.load_state_dict(checkpoint['phrase_model_optimizer'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimD.load_state_dict(checkpoint['discriminator_optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name, is_best=False):
        gpu_cnt = len(self.config.gpu_device)
        file_name = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)

        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.module.state_dict() if gpu_cnt > 1 else self.model.state_dict(),
            'model_optimizer': self.optimVAE.state_dict(),
            'phrase_model_state_dict': self.phrase_model.module.state_dict() if gpu_cnt > 1 else self.phrase_model.state_dict(),
            'phrase_model_optimizer': self.optim_phrase.state_dict(),
            'discriminator_state_dict': self.discriminator.module.state_dict() if gpu_cnt > 1 else self.discriminator.state_dict(),
            'discriminator_optimizer': self.optimD.state_dict()
        }

        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name,
                            os.path.join(self.config.root_path, self.config.checkpoint_dir, 'model_best.pth.tar'))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.current_epoch, self.config.epoch):
            self.current_epoch = epoch
            is_best = self.train_one_epoch()
            self.save_checkpoint(self.config.checkpoint_file, is_best)
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimVAE, mode='min', factor=0.8, cooldown=4)
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_phrase, mode='min', factor=0.8, cooldown=4)
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimD, mode='min', factor=0.8, cooldown=4)

    def train_one_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        self.model.train()
        self.phrase_model.train()
        self.discriminator.train()

        epoch_loss = AverageMeter()
        epoch_phrase_loss = AverageMeter()
        epoch_lossD = AverageMeter()

        for curr_it, (note, pre_note, pre_phrase, position) in enumerate(tqdm_batch):
            if self.cuda:
                note = note.cuda(async=self.config.async_loading)
                pre_note = pre_note.cuda(async=self.config.async_loading)
                pre_phrase = pre_phrase.cuda(async=self.config.async_loading)
                position = position.cuda(async=self.config.async_loading)

            note = Variable(note)
            pre_note = Variable(pre_note)
            pre_phrase = Variable(pre_phrase)
            position = Variable(position)

            ####################
            self.model.zero_grad()
            self.phrase_model.zero_grad()
            self.discriminator.zero_grad()
            zeros = torch.randn(note.size(0), ).fill_(0.).cuda()
            ones = torch.randn(note.size(0), ).fill_(1.).cuda()

            #################### Generator ####################
            self.free(self.model)
            self.frozen(self.phrase_model)
            self.frozen(self.discriminator)

            phrase_feature, _, _ = self.phrase_model(pre_phrase, position)
            gen_note, mean, var, pre_mean, pre_var = self.model(note, pre_note, phrase_feature)
            f_logits = self.discriminator(gen_note)

            gen_loss = self.loss(gen_note, note, f_logits, mean, var, pre_mean, pre_var)
            gen_loss.backward(retain_graph=True)
            self.optimVAE.step()

            #################### Phrase Encoder ####################
            self.free(self.phrase_model)
            self.frozen(self.model)
            self.frozen(self.discriminator)

            phrase_feature, mean, var = self.phrase_model(pre_phrase, position)
            gen_note, _, _, _, _ = self.model(note, pre_note, phrase_feature)
            f_logits = self.discriminator(gen_note)

            phrase_loss = self.phrase_loss(gen_note, note, f_logits, mean, var)
            phrase_loss.backward(retain_graph=True)
            self.optim_phrase.step()

            #################### Discriminator Encoder ####################
            self.free(self.phrase_model)
            self.frozen(self.model)
            self.frozen(self.discriminator)

            phrase_feature, _, _ = self.phrase_model(pre_phrase, position)
            gen_note, _, _, _, _ = self.model(note, pre_note, phrase_feature)

            f_logits = self.discriminator(gen_note)
            r_logits = self.discriminator(note)

            disc_loss = self.lossD(f_logits, r_logits)
            disc_loss.backward(retain_graph=True)
            self.optimD.step()

            ####################
            epoch_lossD.update(disc_loss.item())
            epoch_loss.update(gen_loss.item())
            epoch_phrase_loss.update(phrase_loss.item())

            self.current_iteration += 1

            self.summary_writer.add_scalar("epoch/Generator_loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/PhrasseEncoder_loss", epoch_phrase_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/Discriminator_loss", epoch_lossD.val, self.current_iteration)

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Discriminator loss: " +
                         str(epoch_lossD.val) + " - Generator Loss-: " + str(epoch_loss.val))

        if epoch_loss.val < self.best_error:
            self.best_error = epoch_loss.val
            return True
        else:
            return False

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint(self.config.checkpoint_file)
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.root_path, self.config.summary_dir,
                                                                'all_scalars.json'))
        self.summary_writer.close()

