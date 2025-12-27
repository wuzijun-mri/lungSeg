import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .model import Segmentor
from collections import deque


class seg_trainer(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # initialize model
        self.model = Segmentor(opts)
        self.model.cuda()

        # load pretrained MAE model 
        if self.opts.pretrain_path is not None:
            self.model.load_state_dict(torch.load(self.opts.pretrain_path), strict=False)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, betas=opts.betas)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-8)
        
        # track the history of seg_loss
        self.seg_loss_save = deque(maxlen=100)
        self.total_step = 0


    def train_step(self, data, epoch):
        torch.cuda.empty_cache()

        self.total_step += 1

        self.model.train()

        img_src, lbl_src, img_tgt_f1,img_tgt_f2,img_tgt_f3,img_tgt_f4,img_tgt_f5 = \
            data['src_img'].float().cuda(), data['src_lbl'].long().cuda(), data['tgt_f1'].float().cuda(), data['tgt_f2'].float().cuda(), data['tgt_f3'].float().cuda(), data['tgt_f4'].float().cuda(), data['tgt_f5'].float().cuda()
        
        # train on source labeled data
        loss_seg, pred_seg = self.model.forward_train(img_src, lbl_src)


        #======================Total Vairance on five frames=====================
        #-- student forward on all five frames
        pred_f1_logit = self.model.forward(img_tgt_f1)
        pred_f2_logit = self.model.forward(img_tgt_f2)
        pred_f3_logit = self.model.forward(img_tgt_f3)
        pred_f4_logit = self.model.forward(img_tgt_f4)
        pred_f5_logit = self.model.forward(img_tgt_f5)
        #-- convert pred_logit (1, num_cls, 144, 144, 144) to softmax_pred (1,144,144,144)
        pred_f1 = torch.softmax(pred_f1_logit, dim=1)[:,1,:,:,:]
        pred_f2 = torch.softmax(pred_f2_logit, dim=1)[:,1,:,:,:]
        pred_f3 = torch.softmax(pred_f3_logit, dim=1)[:,1,:,:,:]
        pred_f4 = torch.softmax(pred_f4_logit, dim=1)[:,1,:,:,:]
        pred_f5 = torch.softmax(pred_f5_logit, dim=1)[:,1,:,:,:]
        #-- calculate TV loss
        fused_pred = torch.stack((pred_f1, pred_f2, pred_f3, pred_f4, pred_f5), dim=-1)
        fused_img = torch.stack((img_tgt_f1[0], img_tgt_f2[0], img_tgt_f3[0], img_tgt_f4[0], img_tgt_f5[0]), dim=-1).detach()
        mask_diff = torch.abs(torch.diff(fused_pred, dim=-1))
        img_diff = torch.abs(torch.diff(fused_img, dim=-1))
        img_tv_weight = torch.exp(-10 * img_diff)
        weighted_tv = img_tv_weight * mask_diff
        loss_tv = torch.sum(weighted_tv) / (144**3)


        #-- set lambda_tv
        if self.total_step < 25 * 100: # add TC loss after first 25 epochs (2500 steps)
            lambda_tv = 0.0
        else:
            lambda_tv = self.opts.lambda_tc

        # TOTAL LOSS
        loss = (loss_seg) + (lambda_tv * loss_tv)
        
        # loss for print
        self.loss_dict = dict(zip(['loss_seg', 'loss_TC', 'total_loss', 'lambda_TC'],
                                [loss_seg.item(), loss_tv.item(), loss.item(), lambda_tv]))

        # save loss_seg for each iteration
        self.seg_loss_save.append(loss_seg.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_loss_dict(self, epoch, i):
        iter_loss_str = f'(epoch/iter = {epoch}/{i}):  '
        for k, v in self.loss_dict.items():
            iter_loss_str += f'{k}: {v:.3f}  '
        return iter_loss_str


    def scheduler_step(self):
        self.scheduler.step()
        lr_now = self.optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {lr_now}')
    



