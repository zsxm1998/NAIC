import os
import yaml
from types import SimpleNamespace
from collections import OrderedDict

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from trainers.BaseTrainer import BaseTrainer
from datasets.rematch_dataset import RematchDataset, RematchBatchSampler, rematch_collate_fn
from models.embedding import MoCo
from losses.supcon import SupConLoss
from losses.reconstruction import L2ReconstructionLoss, ExpReconstructionLoss, EnlargeReconstructionLoss

class RematchTrainer(BaseTrainer):
    def __init__(self, opt_file='args/rematch_args.yaml'):
        super(RematchTrainer, self).__init__(checkpoint_root='Rematch')

        with open(opt_file) as f:
            opt = yaml.safe_load(f)
            opt = SimpleNamespace(**opt)
            f.seek(0)
            self.logger.info(f'{opt_file} START************************\n'
            f'{f.read()}\n'
            f'************************{opt_file} END**************************\n')
        self.opt = opt

        if opt.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Using device {self.device}')

        self.net = MoCo(34)
        if opt.load_model:
            self.net.load_state_dict(torch.load(opt.load_model, map_location=self.device))
            self.logger.info(f'Model loaded from {opt.load_model}')
        self.net.to(device=self.device)

        self.train_dataset = RematchDataset(opt.source)
        self.n_train = len(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_sampler=RematchBatchSampler(self.train_dataset, opt.batch_size),
                                       collate_fn=rematch_collate_fn,
                                       num_workers=8, 
                                       pin_memory=True)

        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.epochs, eta_min=1e-8)

        self.criterion = SupConLoss()

        self.epochs = opt.epochs
        self.save_cp = opt.save_cp
        self.early_stopping = opt.early_stopping

        self.logger.info(f'''Starting training net:
        Epochs:          {opt.epochs}
        Batch size:      {opt.batch_size}
        Learning rate:   {opt.lr}
        Training size:   {self.n_train}
        Checkpoints:     {opt.save_cp}
        Device:          {self.device.type}
        Data source:     {opt.source}
        Training info:   {opt.info}
    ''')

    def train(self):
        global_step = 0
        best_val_score = float('inf')
        useless_epoch_count = 0
        for epoch in range(self.epochs):
            try:
                self.net.train()
                epoch_loss = 0
                with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='q') as pbar:
                    for in_q, in_k, q_label, k_label in self.train_loader:
                        global_step += 1
                        in_q, in_k, q_label, k_label= in_q.to(self.device), in_k.to(self.device), q_label.to(self.device), k_label.to(self.device)

                        q, k, queue, queue_label = self.net(in_q, in_k, k_label)
                        k, k_label = torch.cat([k, queue]), torch.cat([k_label, queue_label])

                        loss = self.criterion(q, q_label, k, k_label)
                        self.writer.add_scalar('Train_Loss/loss', loss.item(), global_step)
                        epoch_loss += loss.item() * in_q.size(0)
                        pbar.set_postfix(OrderedDict(**{'loss ': loss.item()}))

                        self.optimizer.zero_grad()
                        loss.backward()
                        #nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                        self.optimizer.step()

                        pbar.update(in_q.shape[0])

                self.logger.info(f'Train epoch {epoch + 1} loss: {epoch_loss/self.n_train}')

                for tag, value in self.net.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    if value.grad is not None:
                        self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
                self.scheduler.step()

                if self.save_cp:
                    torch.save(self.net.state_dict(), self.checkpoint_dir + f'Net_epoch{epoch + 1}.pth')
                    self.logger.info(f'Checkpoint {epoch + 1} saved !')
                else:
                    torch.save(self.net.state_dict(), self.checkpoint_dir + 'Net_last.pth')
                    self.logger.info('Last model saved !')

                if epoch_loss < best_val_score:
                    best_val_score = epoch_loss
                    torch.save(self.net.state_dict(), self.checkpoint_dir + 'Net_best.pth')
                    self.logger.info('Best model saved !')
                    useless_epoch_count = 0
                else:
                    useless_epoch_count += 1

                if self.early_stopping and useless_epoch_count == self.early_stopping:
                    self.logger.info(f'There are {useless_epoch_count} useless epochs! Early Stop Training!')
                    break

            except KeyboardInterrupt:
                self.logger.info('Receive KeyboardInterrupt, stop training...')
                break

        if not self.save_cp:
            torch.save(self.net.state_dict(), self.checkpoint_dir + 'Net_last.pth')
            self.logger.info('Last model saved !')

    def __del__(self):
        super(RematchTrainer, self).__del__()
