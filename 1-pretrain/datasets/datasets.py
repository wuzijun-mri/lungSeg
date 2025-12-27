import os
import torchio as tio
import torch
import numpy as np
import random
import nibabel as nib


class mae_dataset(torch.utils.data.Dataset):

    def __init__(self, opts):
        self.opts = opts

        """get all CT (MRI) image full path stored in the CT (MRI) directory"""
        self.path_list = []

        src_dir = opts.src_dir
        tgt_dir = opts.tgt_dir
        
        num_src = 0
        for filename in os.listdir(src_dir):
            path = os.path.join(src_dir, filename)
            self.path_list.append(path)
            num_src += 1

        num_tgt = 0
        for filename in os.listdir(tgt_dir):
            if '_f3.nii.gz' not in filename: continue
            path = os.path.join(tgt_dir, filename)
            self.path_list.append(path)
            num_tgt += 1

        print(f'# MRI images = {str(num_tgt)}.')
        print(f'# CT images = {str(num_src)}.')
        print(f'# all images = {len(self.path_list)}')


    def __getitem__(self, index):
        
        img_path = random.choice(self.path_list)
        tmp_scans = nib.load(img_path)
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        tmp_scans = torch.from_numpy(tmp_scans) #(x,y,z)

        px, py, pz = self.opts.patch_size
        x1, y1, z1 = tmp_scans.shape

        # data augmentataion (scaling, rotation)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.unsqueeze(0)))
        transforms = tio.RandomAffine(
            p=self.opts.aug_prob,
            scales=(0.75, 1.25), 
            degrees=30,
            isotropic=True,
            default_pad_value=0,
            image_interpolation='linear'
        )
        sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float() #(1,x,y,z)

        # randomly select a patch
        if x1 - px == 0:
            x_idx = 0
        else:
            x_idx = np.random.randint(0, x1 - px)
        if y1 - py == 0:
            y_idx = 0
        else:
            y_idx = np.random.randint(0, y1 - py)
        if z1 - pz == 0:
            z_idx = 0
        else:
            z_idx = np.random.randint(0, z1 - pz)

        image_patch = tmp_scans[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1,144,144,144)

        return image_patch


    def __len__(self):
        return 1000


class seg_dataset(torch.utils.data.Dataset):

    def __init__(self, opts):
        self.opts = opts
        
        """get all CT (MRI) image full path stored in the CT (MRI) directory"""
        self.src_path_list = []
        self.tgt_path_list = []

        src_dir = opts.src_dir
        tgt_dir = opts.tgt_dir

        for filename in os.listdir(src_dir):
            path = os.path.join(src_dir, filename)
            self.src_path_list.append(path)

        for filename in os.listdir(tgt_dir):
            if '_f3.nii.gz' not in filename: continue
            path = os.path.join(tgt_dir, filename)
            self.tgt_path_list.append(path)

        print(f'# MRI images = {len(self.tgt_path_list)}.')
        print(f'# CT images = {len(self.src_path_list)}.')


    def __getitem__(self, index):


        """ unlabeled target data """

        # randomly choose a target image full path
        img_path_f3 = random.choice(self.tgt_path_list)

        # all five frames
        tmp_scans_f1 = nib.load(img_path_f3.replace('_f3','_f1'))
        tmp_scans_f2 = nib.load(img_path_f3.replace('_f3','_f2'))
        tmp_scans_f3 = nib.load(img_path_f3)
        tmp_scans_f4 = nib.load(img_path_f3.replace('_f3','_f4'))
        tmp_scans_f5 = nib.load(img_path_f3.replace('_f3','_f5'))
        tmp_scans_f1 = np.squeeze(tmp_scans_f1.get_fdata())
        tmp_scans_f2 = np.squeeze(tmp_scans_f2.get_fdata())
        tmp_scans_f3 = np.squeeze(tmp_scans_f3.get_fdata())
        tmp_scans_f4 = np.squeeze(tmp_scans_f4.get_fdata())
        tmp_scans_f5 = np.squeeze(tmp_scans_f5.get_fdata())

        tmp_scans_f1 = torch.from_numpy(tmp_scans_f1)
        tmp_scans_f2 = torch.from_numpy(tmp_scans_f2)
        tmp_scans_f3 = torch.from_numpy(tmp_scans_f3)
        tmp_scans_f4 = torch.from_numpy(tmp_scans_f4)
        tmp_scans_f5 = torch.from_numpy(tmp_scans_f5)

        px, py, pz = self.opts.patch_size
        x1, y1, z1 = tmp_scans_f1.shape

        # augmentation
        sbj_f1 = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans_f1.unsqueeze(0)))
        sbj_f2 = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans_f2.unsqueeze(0)))
        sbj_f3 = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans_f3.unsqueeze(0)))
        sbj_f4 = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans_f4.unsqueeze(0)))
        sbj_f5 = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans_f5.unsqueeze(0)))
        # data augmentation at opt.aug_prob
        if np.random.uniform() <= self.opts.aug_prob: 
            scales_thisTime = np.random.uniform(low=0.75, high=1.25)
            degrees_thisTime = np.random.uniform(low=-20, high=20)
            transforms = tio.RandomAffine(
                scales=(scales_thisTime, scales_thisTime), 
                degrees=(degrees_thisTime, degrees_thisTime),
                isotropic=True,
                default_pad_value=0,
                image_interpolation='linear'
            )
            sbj_f1 = transforms(sbj_f1)
            sbj_f2 = transforms(sbj_f2)
            sbj_f3 = transforms(sbj_f3)
            sbj_f4 = transforms(sbj_f4)
            sbj_f5 = transforms(sbj_f5)
        tmp_scans_f1 = sbj_f1['one_image'].data.float()
        tmp_scans_f2 = sbj_f2['one_image'].data.float()
        tmp_scans_f3 = sbj_f3['one_image'].data.float()
        tmp_scans_f4 = sbj_f4['one_image'].data.float()
        tmp_scans_f5 = sbj_f5['one_image'].data.float()

        # randomly select a patch
        if x1 - px == 0:
            x_idx = 0
        else:
            x_idx = np.random.randint(0, x1 - px)
        if y1 - py == 0:
            y_idx = 0
        else:
            y_idx = np.random.randint(0, y1 - py)
        if z1 - pz == 0:
            z_idx = 0
        else:
            z_idx = np.random.randint(0, z1 - pz)

        tgt_f1 =  tmp_scans_f1[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144)
        tgt_f2 =  tmp_scans_f2[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144)
        tgt_f3 =  tmp_scans_f3[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144)
        tgt_f4 =  tmp_scans_f4[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144)
        tgt_f5 =  tmp_scans_f5[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144)




        ''' labeled source data '''

        img_path = random.choice(self.src_path_list)
        label_path = img_path.replace('img','label') # corresponding label path

        tmp_scans = nib.load(img_path)
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        tmp_label = nib.load(label_path)
        tmp_label = np.squeeze(tmp_label.get_fdata())

        tmp_scans = torch.from_numpy(tmp_scans)
        tmp_label = torch.from_numpy(tmp_label) #(x,y)

        px, py, pz = self.opts.patch_size
        x1, y1, z1 = tmp_scans.shape

        # data augmentation
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.unsqueeze(0)),
                          a_segmentation=tio.LabelMap(tensor=tmp_label.unsqueeze(0)))
        transforms = tio.Compose([tio.RandomAffine(p=self.opts.aug_prob,
                                                   scales=(0.75, 1.25), 
                                                   degrees=30,
                                                   isotropic=True, 
                                                   default_pad_value=0,
                                                   image_interpolation='linear',
                                                   label_interpolation='nearest'),
                                tio.RandomBiasField(p=self.opts.aug_prob),
                                tio.RandomGamma(p=self.opts.aug_prob, log_gamma=(-0.3, 0.3))
                                ])
        sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        # randomly select a patch
        if x1 - px == 0:
            x_idx = 0
        else:
            x_idx = np.random.randint(0, x1 - px)
        if y1 - py == 0:
            y_idx = 0
        else:
            y_idx = np.random.randint(0, y1 - py)
        if z1 - pz == 0:
            z_idx = 0
        else:
            z_idx = np.random.randint(0, z1 - pz)

        src_img = tmp_scans[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz] #(1, 144, 144, 144)
        src_lbl = torch.squeeze(tmp_label[:, x_idx:x_idx + px, y_idx:y_idx + py, z_idx:z_idx + pz]) #(144, 144, 144)


        data_dict = {'src_img': src_img,
                      'src_lbl': src_lbl,
                      'tgt_f1': tgt_f1,
                      'tgt_f2': tgt_f2,
                      'tgt_f3': tgt_f3,
                      'tgt_f4': tgt_f4,
                      'tgt_f5': tgt_f5,}

        return data_dict


    def __len__(self):
        return 100

