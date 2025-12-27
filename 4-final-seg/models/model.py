import torch
import torch.nn as nn
import numpy as np
from .buildingblocks import create_encoders, ExtResNetBlock, DeepLabHead
from .loss import DC_and_CE_loss


class Segmentor(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # encoder (may load pretrain encoder)
        self.encoder = create_encoders(in_channels=1, f_maps=(512,512,512,512,512,512,512,512), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)
        # decoder
        self.seg_decoder = DeepLabHead(in_channels=512, aspp_channel=256, num_classes=2, ratio=4)
        # CEloss
        self.CE = nn.CrossEntropyLoss()

    def seg_loss(self, label, pred):
        # this version has confidence mask
        loss = DC_and_CE_loss( {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss_seg = loss(pred, label)
        return loss_seg

    # only for student warm-up in source domain
    def forward_train(self, img, lbl):
        # encoder
        x = img
        for f in self.encoder:
            x = f(x)
        # ASPP decoder
        pred = self.seg_decoder(x)
        # loss
        loss = self.seg_loss(lbl, pred)
        return loss, pred

    def forward(self, img):
        # encoder
        x = img
        for f in self.encoder:
            x = f(x)
        # ASPP decoder
        pred = self.seg_decoder(x)
        return pred
