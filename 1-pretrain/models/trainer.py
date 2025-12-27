import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .model import MAE


class mae_trainer(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        self.model = MAE(opts)
        self.model.cuda()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, betas=opts.betas)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)
        

    def train_step(self, data, epoch):

        self.model.train()
        
        loss, pred, mask = self.model.forward_train(data.float().cuda())
        
        self.loss_dict = dict(zip(['loss_mse'], [loss.item()]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def scheduler_step(self):
        self.scheduler.step()
        lr_now = self.optimizer.param_groups[0]['lr']
        print(f'Current lr: {lr_now}')
        

    def get_loss_dict(self, epoch, i):
        iter_loss_str = f'(epoch/iter = {epoch}/{i}) '
        for k, v in self.loss_dict.items():
            iter_loss_str += f'{k}: {v:.3f}  '
        return iter_loss_str




