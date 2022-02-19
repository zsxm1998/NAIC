import os
from numpy import dtype
import yaml
from types import SimpleNamespace
from collections import OrderedDict

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from trainers.BaseTrainer import BaseTrainer
from datasets.reconstruction_dataset import ReconstructionDataset
from models.autoencoder import AutoEncoderMLP
from models.conv import AutoEncoderConv
from losses.reconstruction import NormalizeReconstructionLoss, CosineReconstructionLoss, L2ReconstructionLoss, EnlargeReconstructionLoss

class ReconstructionTrainer(BaseTrainer):
    def __init__(self, opt_file='args/reconstruction_args.yaml'):
        super(ReconstructionTrainer, self).__init__(checkpoint_root='AutoEncoder')

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

        if opt.model.lower() == 'mlp':
            self.net = AutoEncoderMLP(opt.intermediate_dim, opt.input_dim, opt.output_dim)
        elif opt.model.lower() == 'conv':
            self.net = AutoEncoderConv(opt.intermediate_dim, opt.input_dim, opt.output_dim)
        else:
            raise ValueError('"model" should be either mlp or conv')
        
        if opt.load_encoder:
            self.net.encoder.load_state_dict(torch.load(opt.load_encoder, map_location=self.device))
            self.logger.info(f'Encoder loaded from {opt.load_encoder}')
        if opt.load_decoder:
            self.net.decoder.load_state_dict(torch.load(opt.load_decoder, map_location=self.device))
            self.logger.info(f'Decoder loaded from {opt.load_decoder}')
        self.net.to(device=self.device)

        self.train_dataset = ReconstructionDataset(opt.source, 'train_data_reshape.pt')
        self.n_train = len(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=8, 
                                       pin_memory=True)

        self.val_dataset = ReconstructionDataset(opt.source, 'val_data_reshape.pt')
        self.n_val = len(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=8, 
                                     pin_memory=True)

        self.optimizer = optim.RMSprop(self.net.parameters(), lr=opt.lr)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.epochs, eta_min=1e-9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=50)

        self.criterion = nn.MSELoss()
        self.val_criterion = L2ReconstructionLoss()

        self.epochs = opt.epochs
        self.save_cp = opt.save_cp
        self.early_stopping = opt.early_stopping

        self.logger.info(f'''Starting training net:
        Epochs:          {opt.epochs}
        Batch size:      {opt.batch_size}
        Learning rate:   {opt.lr}
        Training size:   {self.n_train}
        Validation size: {self.n_val}
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
                with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='vector') as pbar:
                    for vectors in self.train_loader:
                        global_step += 1
                        vectors = vectors.to(self.device)

                        reconstruct_v = self.net(vectors if self.opt.model.lower() == 'mlp' else vectors.unsqueeze(1))

                        loss = self.criterion(reconstruct_v, vectors)

                        self.writer.add_scalar('Train_Loss/loss', loss.item(), global_step)
                        epoch_loss += loss.item() * vectors.size(0)
                        pbar.set_postfix(OrderedDict(**{'loss (batch)': loss.item()}))

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                        self.optimizer.step()

                        pbar.update(vectors.shape[0])

                epoch_loss /= self.n_train
                self.logger.info(f'Train epoch {epoch + 1} loss: {epoch_loss}')

                for tag, value in self.net.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    if value.grad is not None:
                        self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)

                val_score = self.evaluate()
                self.logger.info(f'Validation epoch {epoch + 1} loss: {val_score}')
                self.scheduler.step(val_score)
                self.writer.add_scalar('Val_Loss/loss', val_score, global_step)

                if self.save_cp:
                    torch.save(self.net.encoder.state_dict(), self.checkpoint_dir + f'Encoder_epoch{epoch + 1}.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_epoch{epoch + 1}.pth')
                    self.logger.info(f'Checkpoint {epoch + 1} saved !')
                else:
                    torch.save(self.net.encoder.state_dict(), self.checkpoint_dir + f'Encoder_last.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_last.pth')
                    self.logger.info('Last model saved !')

                if val_score < best_val_score:
                    best_val_score = val_score
                    torch.save(self.net.encoder.state_dict(), self.checkpoint_dir + f'Encoder_best.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_best.pth')
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
            torch.save(self.net.encoder.state_dict(), self.checkpoint_dir + f'Encoder_last.pth')
            torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_last.pth')
            self.logger.info('Last model saved !')

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        val_score = 0
        with tqdm(total=self.n_val, desc=f'Validation round', unit='vector', leave=False) as pbar:
            for vectors in self.val_loader:
                vectors = vectors.to(self.device)
                reconstruct_v = self.net(vectors if self.opt.model.lower() == 'mlp' else vectors.unsqueeze(1))

                loss = self.val_criterion(vectors, reconstruct_v)
                val_score += loss.item() * vectors.size(0)
                pbar.set_postfix(OrderedDict(**{'loss (batch)': loss.item()}))
                pbar.update(vectors.shape[0])
        
        val_score /= self.n_val
        self.net.train()
        return val_score

    def __del__(self):
        super(ReconstructionTrainer, self).__del__()
