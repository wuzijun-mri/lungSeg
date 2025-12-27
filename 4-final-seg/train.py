import os, time, copy
import numpy as np
import torch
from models.trainer import seg_trainer
from datasets.datasets import seg_dataset
import argparse
import random
import logging
from logging.handlers import RotatingFileHandler


def set_random_seed(seed): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def setup_logging(log_dir, filename):
    log_file = os.path.join(log_dir, f'{filename}.log')
    
    logger = logging.getLogger('MRI_Training')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def prepare_dir(ndir):
    is_exists = os.path.exists(ndir)
    if not is_exists:
        os.makedirs(ndir)


# training configurations

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ckpt_dir', type=str, default='PATH_TO_CHECKPOINT')

parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--annealing_epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save_per_epoch', type=int, default=1)
parser.add_argument('--print_per_iter', type=int, default=10)
parser.add_argument('--lambda_tc', type=float, default=0.5)

parser.add_argument('--src_dir', type=str, default='PATH_TO_MRI')
parser.add_argument('--tgt_dir', type=str, default='PATH_TO_MRI')

parser.add_argument('--pretrain_path', type=str, default="PATH_TO_PRETRAINED_MAE")

parser.add_argument('--cls_num', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--aug_prob', type=int, default=0.25)
parser.add_argument('--patch_size', type=tuple, default=(144,144,144))
parser.add_argument('--n_workers', type=int, default=8)

opts = parser.parse_args()

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    set_random_seed(opts.seed)

    # assign GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)

    # make checkpoint directory
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    model_name = f'initial-seg-{time_now}'
    ckpt_dir = os.path.join(opts.ckpt_dir, model_name)
    prepare_dir(ckpt_dir)

    # print configuration
    logger = setup_logging(ckpt_dir, 'train')
    logger.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    logger.info("=" * 80)
    logger.info(f"Training parameters:")
    logger.info(f"  - GPU: {opts.gpu_id}")
    logger.info(f"  - Number of iterations: {opts.epoch}")
    logger.info(f"  - Number of annealing iterations: {opts.annealing_epoch}")
    logger.info(f"  - Patch size: {opts.patch_size}")
    logger.info(f"  - Save per epoch: {opts.save_per_epoch}")
    logger.info(f"  - Print per iter: {opts.print_per_iter}")

    # load trainer
    trainer = seg_trainer(opts)

    # load data
    dataset = seg_dataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opts.batch_size,
                                                shuffle=True,
                                                num_workers=opts.n_workers)

    prev_MA = None
    cnt = 0
    early_stop = False

    for epoch in range(opts.epoch + opts.annealing_epoch + 1): # each epoch

        for i, data in enumerate(train_loader): # 100 sample / 1 batchsize = 100 steps
            
            trainer.train_step(data, epoch)

            if i % opts.print_per_iter == 0:
                loss_str = trainer.get_loss_dict(epoch, i)
                logger.info(loss_str)

            # early stop
            seg_loss_list = trainer.seg_loss_save
            if len(seg_loss_list) == 100:
                # Moving average of last 100 iter
                loss_MA = sum(seg_loss_list) / 100.0

                if prev_MA is not None:
                    if loss_MA < prev_MA:
                        cnt = 0
                    else:
                        cnt += 1

                prev_MA = loss_MA

                if cnt >= 10: # early stop criteria met
                    early_stop = True
                    logger.info(f"Early stopping triggered at epoch {epoch}, iter {i}")

        if early_stop:
            break

        # save checkpoint for completed epochs
        if epoch % opts.save_per_epoch == 0:
            torch.save(trainer.model.state_dict(), os.path.join(ckpt_dir, f'model-epoch-{epoch}.pth'))

        # annealing epoch scheduler
        if epoch > opts.epoch:
            trainer.scheduler_step()

            

