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
from graphs.models.wae_v1.model import Model
from graphs.models.discriminator import Discriminator
from graphs.models.discriminator_z import DiscriminatorZ
from graphs.losses.loss import WAELoss, DLoss
from data.noteDataset import NoteDataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate

cudnn.benchmark = True


class MCWAE(object):
    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("MC_WAE")

        self.batch_size = self.config.batch_size

        # define models ( generator and discriminator)
        self.model = Model()
        self.discriminator = Discriminator()
        self.discriminator_z = DiscriminatorZ()

        # define dataloader
        self.dataset = NoteDataset(self.config.root_path, self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=3,
                                     pin_memory=self.config.pin_memory, collate_fn=self.make_batch)

        # define loss
        self.loss = WAELoss()
        self.lossD = DLoss()

        # define optimizers for both generator and discriminator
        self.lr = self.config.learning_rate
        self.lrD = self.config.learning_rate
        self.lrDZ = self.config.learning_rate
        self.optimWAE = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lrD)
        self.optimDZ = torch.optim.Adam(self.discriminator_z.parameters(), lr=self.lrDZ)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_error = 9999999999.

        self.fixed_noise = Variable(torch.randn(1, 510, dtype=torch.float32))
        self.zero_note = Variable(torch.zeros(1, 1, 384, 96, dtype=torch.float32))

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
            self.fixed_noise = self.fixed_noise.cuda(async=self.config.async_loading)
            self.zero_note = self.zero_note.cuda(async=self.config.async_loading)
            self.device = torch.device("cuda")

        else:
            self.logger.info("Program will run on *****CPU***** ")
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.discriminator_z = self.discriminator_z.to(self.device)
        self.loss = self.loss.to(self.device)
        self.lossD = self.lossD.to(self.device)

        if len(self.config.gpu_device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.config.gpu_device)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.config.gpu_device)
            self.discriminator_z = nn.DataParallel(self.discriminator_z, device_ids=self.config.gpu_device)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='MC_WAE')

    def free(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def frozen(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def make_batch(self, samples):
        note = np.concatenate([sample['note'] for sample in samples], axis=0)
        pre_note = np.concatenate([sample['pre_note'] for sample in samples], axis=0)
        position = np.concatenate([sample['position'] for sample in samples], axis=0)

        return tuple([torch.tensor(note, dtype=torch.float), torch.tensor(pre_note, dtype=torch.float),
                      torch.tensor(position, dtype=torch.long)])

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimWAE.load_state_dict(checkpoint['model_optimizer'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimD.load_state_dict(checkpoint['discriminator_optimizer'])
            self.discriminator_z.load_state_dict(checkpoint['discriminatorZ_state_dict'])
            self.optimDZ.load_state_dict(checkpoint['discriminatorZ_optimizer'])
            self.fixed_noise = checkpoint['fixed_noise']
            self.manual_seed = checkpoint['manual_seed']

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
            'model_optimizer': self.optimWAE.state_dict(),
            'discriminator_state_dict': self.discriminator.module.state_dict() if gpu_cnt > 1 else self.discriminator.state_dict(),
            'discriminator_optimizer': self.optimD.state_dict(),
            'discriminatorZ_state_dict': self.discriminator_z.module.state_dict() if gpu_cnt > 1 else self.discriminator_z.state_dict(),
            'discriminatorZ_optimizer': self.optimDZ.state_dict(),
            'fixed_noise': self.fixed_noise,
            'manual_seed': self.manual_seed
        }

        # Save the state
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
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimWAE, mode='min', factor=0.8, cooldown=4)
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimD, mode='min', factor=0.8, cooldown=4)
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimDZ, mode='min', factor=0.8, cooldown=4)

    def train_one_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.dataset.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        self.model.train()
        self.discriminator.train()
        self.discriminator_z.train()

        epoch_loss = AverageMeter()
        epoch_lossD = AverageMeter()
        epoch_lossDZ = AverageMeter()

        for curr_it, (note, pre_note, position) in enumerate(tqdm_batch):
            if self.cuda:
                note = note.cuda(async=self.config.async_loading)
                pre_note = pre_note.cuda(async=self.config.async_loading)
                position = position.cuda(async=self.config.async_loading)

            ####################
            note = Variable(note)
            pre_note = Variable(pre_note)
            position = Variable(position)

            self.model.zero_grad()
            self.discriminator.zero_grad()
            self.discriminator_z.zero_grad()
            zeros = torch.randn(note.size(0), ).fill_(0.).cuda()
            ones = torch.randn(note.size(0), ).fill_(1.).cuda()

            #################### Generator ####################
            self.free(self.model)
            self.frozen(self.discriminator)
            self.frozen(self.discriminator_z)

            gen_note, z = self.model(note, pre_note, position)
            f_logits = self.discriminator(gen_note)
            fz_logits = self.discriminator_z(z)

            loss_model = self.loss(gen_note, note, f_logits, fz_logits)
            loss_model.backward(retain_graph=True)
            self.optimWAE.step()

            #################### Discriminator ####################
            self.free(self.discriminator)
            self.frozen(self.discriminator_z)
            self.frozen(self.model)

            r_logits = self.discriminator(note)
            f_logits = self.discriminator(self.model(note, pre_note, position)[0])

            loss_D = -((torch.log(1.001 - r_logits).mean()) + torch.log(f_logits).mean())
            loss_D.backward(retain_graph=True)
            self.optimD.step()

            #################### DiscriminatorZ ####################
            self.free(self.discriminator_z)
            self.frozen(self.discriminator)
            self.frozen(self.model)

            z_fake = torch.randn(note.size()[0], 510)
            z_fake = z_fake.cuda()

            r_logits = self.discriminator_z(self.model(note, pre_note, position)[1])
            f_logits = self.discriminator_z(z_fake)

            loss_DZ = -((torch.log(1.001 - r_logits).mean()) + torch.log(f_logits).mean())
            loss_DZ.backward(retain_graph=True)
            self.optimDZ.step()

            ####################
            epoch_lossD.update(loss_D.item())
            epoch_lossDZ.update(loss_DZ.item())
            epoch_loss.update(loss_model.item())

            self.current_iteration += 1

            self.summary_writer.add_scalar("epoch/Generator_loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/Discriminator_loss", epoch_lossD.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/DiscriminatorZ_loss", epoch_lossDZ.val, self.current_iteration)

        out_img = self.model(self.fixed_noise, self.zero_note, torch.tensor([330], dtype=torch.long).cuda(), False)
        self.summary_writer.add_image('train/generated_image',
                                      torch.gt(out_img, 0.3).type('torch.FloatTensor').view(1, 384, 96) * 255,
                                      self.current_iteration)

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


