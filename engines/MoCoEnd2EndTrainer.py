import os
import yaml
from types import SimpleNamespace
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from .BaseTrainer import BaseTrainer
from datasets.rematch_dataset import RematchDataset, RematchBatchSampler, rematch_collate_fn, RematchEvalDataset
from models.mocoend2end import MoCo
from losses.supcon import SupConLoss
from losses.reconstruction import L2ReconstructionLoss, ExpReconstructionLoss, EnlargeReconstructionLoss
from utils.CosineAnnealingWithWarmUpLR import CosineAnnealingWithWarmUpLR

class End2EndTrainer(BaseTrainer):
    def __init__(self, opt_file='args/moco_end2end_args.yaml'):
        super(End2EndTrainer, self).__init__(checkpoint_root='End2End')

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

        self.train_dataset = RematchDataset(opt.source, batch_size=opt.batch_size)
        self.n_train = len(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_sampler=RematchBatchSampler(self.train_dataset, opt.batch_size),
                                       collate_fn=rematch_collate_fn,
                                       num_workers=8, 
                                       pin_memory=True)

        self.val_dataset = RematchEvalDataset(opt.source)
        self.n_val = len(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=8, 
                                     pin_memory=True)

        self.net = MoCo(opt.model_depth, id_num=len(self.train_dataset.datas), extractor_out_dim=opt.feature_dim, compress_dim=int(opt.byte_rate//2))
        if opt.load_model:
            self.net.load_state_dict(torch.load(opt.load_model, map_location=self.device))
            self.logger.info(f'Model loaded from {opt.load_model}')
        self.net.to(device=self.device)

        # self.optimizer = optim.Adam([{'params':self.net.extractor_q.parameters()}, 
        #                              {'params':list(self.net.encoder_q.parameters())+list(self.net.decoder.parameters()), 'lr':0.001}], lr=opt.lr)
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.epochs, eta_min=1e-8)
        self.scheduler = CosineAnnealingWithWarmUpLR(self.optimizer, T_total=opt.epochs, eta_min=1e-8, warm_up_lr=opt.lr/100, warm_up_step=opt.warm_up_step)

        self.criterion_c = SupConLoss(concat=False)
        self.criterion_r = L2ReconstructionLoss()
        self.criterion_i = nn.CrossEntropyLoss()

        self.epochs = opt.epochs
        self.save_cp = opt.save_cp
        self.early_stopping = opt.early_stopping
        self.training_info = opt.info

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
        best_val_score = -1 #float('inf')
        useless_epoch_count = 0
        for epoch in range(self.epochs):
            try:
                self.net.train()
                epoch_c_loss, epoch_r_loss, r_loss_count, epoch_i_loss, i_loss_count = 0, 0, 0, 0, 0
                with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='q') as pbar:
                    for q_in, k_in, q_label, k_label in self.train_loader:
                        global_step += 1
                        q_in, k_in, q_label, k_label= q_in.to(self.device), k_in.to(self.device), q_label.to(self.device), k_label.to(self.device)

                        #q, k, q_comp, k_comp, queue, queue_label, q_reco, k_reco = self.net(q_in, k_in, k_label)
                        q, k, queue, queue_label, q_reco, k_reco, q_id, q_reco_id, k_reco_id = self.net(q_in, k_in, k_label)

                        c_loss = self.criterion_c(q, q_label, torch.cat([k, queue]), torch.cat([k_label, queue_label]))
                        r_loss = self.criterion_r(torch.cat([q_reco, k_reco]), torch.cat([q, k]))
                        i_loss = self.criterion_i(torch.cat([q_id, q_reco_id, k_reco_id]), torch.cat([q_label, q_label, k_label]))

                        loss = c_loss + r_loss + i_loss

                        self.writer.add_scalar('Train_Loss/contrastive_loss', c_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/reconstruction_loss', r_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/identity_loss', i_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/loss', loss.item(), global_step)
                        epoch_c_loss += c_loss.item() * q_in.size(0)
                        epoch_r_loss += r_loss.item() * (q_in.size(0)+k_in.size(0))
                        epoch_i_loss += i_loss.item() * (q_in.size(0)*2+k_in.size(0))
                        r_loss_count += q_in.size(0) + k_in.size(0)
                        i_loss_count += q_in.size(0) * 2 + k_in.size(0)
                        pbar.set_postfix(OrderedDict(**{'loss (batch)': loss.item(), 'c_loss':c_loss.item(), 'r_loss': r_loss.item(), 'i_loss': i_loss.item()}))

                        self.optimizer.zero_grad()
                        loss.backward()
                        # nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                        self.optimizer.step()

                        pbar.update(q_in.shape[0])

                epoch_c_loss /= self.n_train
                epoch_r_loss /= r_loss_count
                epoch_i_loss /= i_loss_count
                epoch_loss = epoch_c_loss + epoch_r_loss + epoch_i_loss
                self.logger.info(f'Train epoch {epoch+1} loss: {epoch_loss}, contrastive loss: {epoch_c_loss}, reconstruction loss: {epoch_r_loss}, identity loss: {epoch_i_loss}')

                for tag, value in self.net.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    if value.grad is not None:
                        self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)

                mAP, reconstruction_loss = self.evaluate()
                val_score = 0.5 * mAP + 0.5 * np.exp(-reconstruction_loss)
                self.logger.info(f'Train epoch {epoch+1} val_score: {val_score}, mAP: {mAP}, reconstruction_loss:{reconstruction_loss}')
                self.writer.add_scalar('Val/val_score', val_score, global_step)
                self.writer.add_scalar('Val/mAP', mAP, global_step)
                self.writer.add_scalar('Val/reconstruction_loss', reconstruction_loss, global_step)

                self.scheduler.step()

                if self.save_cp:
                    torch.save(self.net.state_dict(), self.checkpoint_dir + f'Net_epoch{epoch + 1}.pth')
                    torch.save(self.net.extractor_q.state_dict(), self.checkpoint_dir + f'Extractor_epoch{epoch + 1}.pth')
                    torch.save(self.net.encoder_q.state_dict(), self.checkpoint_dir + f'Encoder_{self.opt.byte_rate}_epoch{epoch + 1}.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_{self.opt.byte_rate}_epoch{epoch + 1}.pth')
                    self.logger.info(f'Checkpoint {epoch + 1} saved !')
                else:
                    torch.save(self.net.state_dict(), self.checkpoint_dir + 'Net_last.pth')
                    torch.save(self.net.extractor_q.state_dict(), self.checkpoint_dir + 'Extractor_last.pth')
                    torch.save(self.net.encoder_q.state_dict(), self.checkpoint_dir + f'Encoder_{self.opt.byte_rate}_last.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_{self.opt.byte_rate}_last.pth')
                    self.logger.info('Last model saved !')

                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(self.net.state_dict(), self.checkpoint_dir + 'Net_best.pth')
                    torch.save(self.net.extractor_q.state_dict(), self.checkpoint_dir + 'Extractor_best.pth')
                    torch.save(self.net.encoder_q.state_dict(), self.checkpoint_dir + f'Encoder_{self.opt.byte_rate}_best.pth')
                    torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_{self.opt.byte_rate}_best.pth')
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
            torch.save(self.net.extractor_q.state_dict(), self.checkpoint_dir + 'Extractor_last.pth')
            torch.save(self.net.encoder_q.state_dict(), self.checkpoint_dir + f'Encoder_{self.opt.byte_rate}_last.pth')
            torch.save(self.net.decoder.state_dict(), self.checkpoint_dir + f'Decoder_{self.opt.byte_rate}_last.pth')
            self.logger.info('Last model saved !')

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()

        feature_list, label_list = [], []
        reconstruction_loss = 0
        with tqdm(total=self.n_val, desc=f'Validation round', unit='vector', leave=False) as pbar:
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                label_list.append(labels.numpy())
            
                features = self.net.extractor_q(imgs)
                featrues_reco = self.net.decoder(self.net.encoder_q(features).half().float())
                reconstruction_loss += self.criterion_r(featrues_reco, features).item() * labels.shape[0]

                features = F.normalize(features, dim=1).cpu().numpy()
                feature_list.append(features)

                pbar.update(imgs.shape[0])

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        del feature_list, label_list

        dists = np.matmul(features, features.T)
        ranks = np.argsort(-dists, axis=1)

        mAP = 0
        for i, rank in enumerate(ranks):
            ap = 0
            rank = rank[rank!=i]
            label = labels[i]
            rank_label = np.take_along_axis(labels, rank, axis=0)
            correct_rank_idx = np.argwhere(rank_label==label).flatten()
            n_correct = len(correct_rank_idx)
            for j in range(n_correct):
                d_recall = 1 / n_correct
                precision = (j+1) / (correct_rank_idx[j]+1)
                if correct_rank_idx[j] != 0:
                    old_precision = j / correct_rank_idx[j]
                else:
                    old_precision = 1
                ap = ap + d_recall * (old_precision+precision) / 2
            mAP += ap
        
        mAP /= ranks.shape[0]
        reconstruction_loss /= ranks.shape[0]

        self.net.train()
        return mAP, reconstruction_loss

    def __del__(self):
        del self.train_loader, self.val_loader
        super(End2EndTrainer, self).__del__()
