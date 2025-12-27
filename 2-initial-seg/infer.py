import os
import numpy as np
import torch
from models.model import Segmentor
import nibabel as nib
import argparse


def sliding_window_inference_3d( image, patch_size, stride, model, threshold=0.5 ):

    model.eval()

    D, H, W = image.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    # Accumulate probabilities and counts
    prob_sum = np.zeros((2, D, H, W), dtype=np.float32)
    prob_map = np.zeros((2, D, H, W), dtype=np.float32)
    count_map = np.zeros((2, D, H, W), dtype=np.float32)

    # Sliding window inference
    z_starts = [0, D - pz, sz, D - pz]
    y_starts = [0, H - py, sy, H - py]
    x_starts = [0, W - px, sx, W - px]

    for z in z_starts:
        z = min(z, D - pz)
        for y in y_starts:
            y = min(y, H - py)
            for x in x_starts:
                x = min(x, W - px)

                patch = image[z:z+pz, y:y+py, x:x+px]
                patch = torch.from_numpy(patch).float().cuda()
                patch = patch.unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    prob_patch = model(patch) #(1,2,144,144,144)

                prob_sum[:, z:z+pz, y:y+py, x:x+px] += prob_patch[0].cpu().numpy()
                count_map[:, z:z+pz, y:y+py, x:x+px] += 1

    # Avoid division by zero
    mask = count_map > 0
    prob_map[mask] = prob_sum[mask] / count_map[mask]

    # segmentation
    sf = torch.nn.Softmax(dim=0)
    ema_softmax = sf(torch.from_numpy(prob_map))
    pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=0)

    ps_large_p = pseudo_prob.ge(threshold)
    pseudo_label = pseudo_label * ps_large_p.long()

    return pseudo_label.numpy()


def prepare_dir(ndir):
    is_exists = os.path.exists(ndir)
    if not is_exists:
        os.makedirs(ndir)


# COPY training configurations HERE

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ckpt_dir', type=str, default='PATH_TO_SAVE_CHECKPOINT')

parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--annealing_epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save_per_epoch', type=int, default=1)
parser.add_argument('--print_per_iter', type=int, default=10)
parser.add_argument('--lambda_tc', type=float, default=2.0)

parser.add_argument('--src_dir', type=str, default='PATH_TO_CT')
parser.add_argument('--tgt_dir', type=str, default='PATH_TO_MRI')

parser.add_argument('--pretrain_path', type=str, default="PATH_TO_PRETRAINED_MAE")

parser.add_argument('--cls_num', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--aug_prob', type=int, default=0.25)
parser.add_argument('--patch_size', type=tuple, default=(144,144,144))
parser.add_argument('--n_workers', type=int, default=8)

opts = parser.parse_args()


#-- assign GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)

#-- model path
ckpt_dir = 'PATH_TO_CHECKPOINT'
model_name = 'TRAINED_MODEL.pth'
model_path = os.path.join(ckpt_dir, model_name)

#-- infer result directory
infer_dir = os.path.join(ckpt_dir, 'infer')
prepare_dir(infer_dir)

#-- test data set
data_dir = 'PATH_TO_MRI_TRAINING_DATA'

# load trained model
model = Segmentor(opts)
model.cuda()
model.load_state_dict(torch.load(model_path), strict=True)


# start inference
for filename in sorted(os.listdir(data_dir)):
    #-- save path
    save_path = os.path.join(infer_dir, filename)
    print(filename)
    test_img = nib.load(os.path.join(data_dir, filename)).get_fdata()
    test_img = np.squeeze(test_img)
    with torch.no_grad():
        pred_vol = sliding_window_inference_3d(test_img, opts.patch_size, (144-24,144-24,144-24), model, threshold=0.5)
    print(pred_vol.shape)
    pred_vol = pred_vol.astype(np.int8)

    test_img = nib.load(os.path.join(data_dir, filename))
    pred = nib.Nifti1Image(pred_vol, affine=test_img.affine, header=test_img.header)
    nib.save(pred, save_path)
