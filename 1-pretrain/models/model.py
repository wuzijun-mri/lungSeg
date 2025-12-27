import torch
import torch.nn as nn
from .buildingblocks import create_encoders, ExtResNetBlock, res_decoders



class MAE(nn.Module):

    def __init__(self, opts):
        super().__init__()
        # configurations
        self.opts = opts

        # encoder
        self.encoder = create_encoders(in_channels=1, f_maps=(512,512,512,512,512,512,512,512), 
                                       basic_module=ExtResNetBlock, 
                                       conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',num_groups=32)

        # decoder
        self.trans_conv = nn.ConvTranspose3d(in_channels=512, out_channels=32, kernel_size=4, stride=4)
        self.res_decoder = res_decoders(in_channels=32, f_maps=[16],
                                        basic_module=ExtResNetBlock,
                                        conv_kernel_size=3, conv_stride_size=1, conv_padding=0, layer_order='gcr', num_groups=8)
        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=16)
        self.final_conv = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        
        
    def patchify(self, imgs, p): 
        """
        imgs: (N, 1, H, W, D); patch: (p,p,p)
        x: (N, H*W*D/p**3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i//p for i in self.opts.patch_size] 

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p): 
        """
        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i//p for i in self.opts.patch_size]
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p): 
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def recon_loss(self, imgs, pred, mask):
        loss = (pred - imgs) ** 2
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_train(self, image):
        # local_patch
        # mask
        x, mask = self.random_masking(image, self.opts.mask_ratio, self.opts.mask_patch)
        # encoder
        for f in self.encoder:
            x = f(x)
        # decoder
        x = self.trans_conv(x)
        for f in self.res_decoder:
            x = f(x)
        x = self.final_norm(x)
        x = self.final_conv(x)
        pred = torch.sigmoid(x)
        # MSE loss
        loss = self.recon_loss(image, pred, mask)
        return loss, pred, mask

    def forward(self, image): # no loss return
        # local_patch
        # mask
        x, mask = self.random_masking(image, self.opts.mask_ratio, self.opts.mask_patch)
        # encoder
        for f in self.encoder:
            x = f(x)
        # decoder
        x = self.trans_conv(x)
        for f in self.res_decoder:
            x = f(x)
        x = self.final_norm(x)
        x = self.final_conv(x)
        pred = torch.sigmoid(x)
        return pred, mask

