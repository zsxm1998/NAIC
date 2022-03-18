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
import torchvision.transforms as T

from .BaseTrainer import BaseTrainer
from datasets.triplet_dataset import TripletDataset, RandomIdentitySampler
from models.extractor import ExtractorModel
from utils.WarmupMultiStepLR import WarmupMultiStepLR
from losses.triplet import TripletLoss, CenterLoss, CrossEntropyLabelSmooth
from utils.ranger import Ranger
from utils.transforms import RandomPermuteChannel

class ExtractorTrainer(BaseTrainer):
    def __init__(self, opt_file='args/extractor_args.yaml'):
        super(ExtractorTrainer, self).__init__(checkpoint_root='Extractor')

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

        #self.img_mean, self.img_std = (0.5114, 0.4065, 0.4534), (0.1380, 0.0849, 0.0814)
        self.img_mean, self.img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.train_transform = T.Compose([
            T.Resize([256, 128]),
            T.RandomHorizontalFlip(p=0.5),
            #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=self.img_mean, std=self.img_std),
            T.RandomErasing(p=0.5, value=self.img_mean),
        ])
        self.val_transform = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=self.img_mean, std=self.img_std),
        ])

        self.train_dataset = TripletDataset(opt.source, label_file='train_list.txt', transform=self.train_transform)
        self.n_train = len(self.train_dataset)
        self.train_sampler = RandomIdentitySampler(self.train_dataset, opt.batch_size, num_instances=4)
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=self.train_sampler,
                                       batch_size=opt.batch_size,
                                       num_workers=8, 
                                       pin_memory=True)

        self.val_dataset = TripletDataset(opt.source, label_file='val_list.txt', transform=self.val_transform)
        self.n_val = len(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=8, 
                                     pin_memory=True)

        self.net = ExtractorModel(opt.model_name, id_num=len(self.train_dataset.pids), extractor_out_dim=opt.feature_dim)
        if opt.load_model:
            self.net.load_state_dict(torch.load(opt.load_model, map_location=self.device))
            self.logger.info(f'Model loaded from {opt.load_model}')
        self.net.to(device=self.device)
        if torch.cuda.device_count() > 1 and self.device.type != 'cpu':
            self.net = nn.DataParallel(self.net)
            self.logger.info(f'torch.cuda.device_count:{torch.cuda.device_count()}, Use nn.DataParallel')
        self.net_module = self.net.module if isinstance(self.net, nn.DataParallel) else self.net

        self.optimizer = Ranger(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.scheduler = WarmupMultiStepLR(self.optimizer, opt.milestones, opt.gamma, opt.warmup_factor, opt.warmup_iters)
        if opt.load_optimizer:
            self.optimizer.load_state_dict(torch.load(opt.load_optimizer))
            self.logger.info(f'Optimizer loaded from {opt.load_optimizer}')
        if opt.load_scheduler:
            self.scheduler.load_state_dict(torch.load(opt.load_scheduler))
            self.logger.info(f'Scheduler loaded from {opt.load_scheduler}')

        self.criterion_triplet = TripletLoss(margin=0.3)
        self.criterion_center = CenterLoss(num_classes=len(self.train_dataset.pids), feat_dim=opt.feature_dim)
        self.criterion_identity = CrossEntropyLabelSmooth(num_classes=len(self.train_dataset.pids))

        self.epochs = opt.epochs
        self.save_cp = opt.save_cp
        self.early_stopping = opt.early_stopping
        self.eval_on_gpu = opt.eval_on_gpu
        self.training_info = opt.info

        self.logger.info(f'''Starting training net:
        Epochs:          {opt.epochs}
        Batch size:      {opt.batch_size}
        Learning rate:   {opt.lr}
        Training size:   {self.n_train}
        Training pid:    {len(self.train_dataset.pids)}
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
        for epoch in range(self.opt.start_epoch, self.epochs):
            try:
                self.net.train()
                epoch_t_loss, epoch_cen_loss, epoch_i_loss = 0, 0, 0
                epoch_count = 0
                with tqdm(total=len(self.train_sampler), desc=f'Epoch {epoch + 1}/{self.epochs}', unit='img') as pbar:
                    for imgs, labels in self.train_loader:
                        global_step += 1
                        imgs, labels = imgs.to(self.device), labels.to(self.device)

                        ft, fi = self.net(imgs)

                        t_loss = self.criterion_triplet(ft, labels)[0]
                        cen_loss = self.criterion_center(ft, labels)
                        i_loss = self.criterion_identity(fi, labels)

                        loss = t_loss + 0.0005*cen_loss + i_loss
                        
                        self.writer.add_scalar('Train_Loss/triplet_loss', t_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/center_loss', cen_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/identity_loss', i_loss.item(), global_step)
                        self.writer.add_scalar('Train_Loss/Step_Loss', loss.item(), global_step)
                        epoch_t_loss += t_loss.item() * labels.size(0)
                        epoch_cen_loss += cen_loss.item() * labels.size(0)
                        epoch_i_loss += i_loss.item() * labels.size(0)
                        epoch_count += labels.size(0)
                        pbar.set_postfix(OrderedDict(**{'loss': loss.item(),
                                         'triplet': t_loss.item(), 
                                         'center': cen_loss.item(), 
                                         'identity': i_loss.item(),
                                        }))

                        self.optimizer.zero_grad()
                        loss.backward()
                        # nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                        self.optimizer.step()

                        pbar.update(labels.shape[0])

                epoch_t_loss /= epoch_count
                epoch_cen_loss /= epoch_count
                epoch_i_loss /= epoch_count
                epoch_loss = epoch_t_loss + 0.0005*epoch_cen_loss + epoch_i_loss
                self.logger.info(f'Train epoch {epoch+1} loss: {epoch_loss}, '
                                 f'triplet loss: {epoch_t_loss}, '
                                 f'center loss: {epoch_cen_loss}, '
                                 f'identity loss: {epoch_i_loss}'
                                )
                self.writer.add_scalar('Train_Loss/Epoch_Loss', epoch_loss, epoch+1)

                # for tag, value in self.net.named_parameters():
                #     tag = tag.replace('.', '/')
                #     self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #     if value.grad is not None:
                #         self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)

                ACC_reid = self.evaluate()
                self.logger.info(f'Train epoch {epoch+1} ACC_reid: {ACC_reid}')
                self.writer.add_scalar('Val/ACC_reid', ACC_reid, global_step)

                self.scheduler.step()

                if self.save_cp:
                    torch.save(self.net_module.state_dict(), self.checkpoint_dir + f'Net_epoch{epoch + 1}.pth')
                    torch.save(self.net_module.extractor.state_dict(), self.checkpoint_dir + f'Extractor_{self.opt.feature_dim}_epoch{epoch + 1}.pth')
                    self.logger.info(f'Checkpoint {epoch + 1} saved !')
                else:
                    torch.save(self.net_module.state_dict(), self.checkpoint_dir + 'Net_last.pth')
                    torch.save(self.net_module.extractor.state_dict(), self.checkpoint_dir + f'Extractor_{self.opt.feature_dim}_last.pth')
                    self.logger.info('Last model saved !')
                torch.save(self.optimizer.state_dict(), self.checkpoint_dir + 'Optimizer_last.pth')
                torch.save(self.scheduler.state_dict(), self.checkpoint_dir + 'Scheduler_last.pth')

                if ACC_reid > best_val_score:
                    best_val_score = ACC_reid
                    torch.save(self.net_module.state_dict(), self.checkpoint_dir + 'Net_best.pth')
                    torch.save(self.net_module.extractor.state_dict(), self.checkpoint_dir + f'Extractor_{self.opt.feature_dim}_best.pth')
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

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()

        feature_list, label_list = [], []
        with tqdm(total=self.n_val, desc=f'Validation round', unit='img', leave=False) as pbar:
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                label_list.append(labels.numpy())
            
                features, _ = self.net(imgs)

                features = F.normalize(features, dim=1).cpu().numpy()
                feature_list.append(features)

                pbar.update(imgs.shape[0])
        
        labels = np.concatenate(label_list, axis=0)
        features = np.concatenate(feature_list, axis=0)
        del label_list, feature_list
        
        if self.eval_on_gpu:
            try:
                features_t = torch.from_numpy(features).to(self.device)
                dists = torch.mm(features_t, features_t.T)
                ranks = torch.argsort(dists, dim=1, descending=True).cpu().numpy()
                del features_t
            except RuntimeError as e:
                self.eval_on_gpu = False
                self.logger.info(f'Except [{e}], change eval_on_gpu to False')
                dists = np.matmul(features, features.T)
                ranks = np.argsort(-dists, axis=1)
            finally:
                del dists, features
        else:
            dists = np.matmul(features, features.T)
            ranks = np.argsort(-dists, axis=1)
            del dists, features

        acc1, mAP = 0, 0
        for i, rank in enumerate(ranks):
            ap = 0
            rank = rank[rank!=i]
            label = labels[i]
            rank_label = np.take_along_axis(labels, rank, axis=0)
            if rank_label[0] == label:
                acc1 += 1 
            correct_rank_idx = np.argwhere(rank_label==label).flatten()
            correct_rank_idx = correct_rank_idx[correct_rank_idx<100]
            n_correct = len(correct_rank_idx)
            if n_correct > 0:
                d_recall = 1 / n_correct
                for j in range(n_correct):
                    precision = (j+1) / (correct_rank_idx[j]+1)
                    ap += d_recall * precision
            mAP += ap
        
        acc1 /= ranks.shape[0]
        mAP /= ranks.shape[0]
        ACC_reid = (acc1 + mAP) / 2

        self.net.train()
        return ACC_reid

    def __del__(self):
        del self.train_loader, self.val_loader
        super(ExtractorTrainer, self).__del__()
