from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
from tqdm import tqdm 

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import pose.utils.noise as noise
from pose.datasets import surreal_utils
import cv2


class Surreal(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path']  # Root image folder (will contain sub-folders for each scanned model)
        self.is_train = is_train
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']

        # Load annotations
        data = np.load(kwargs['anno_path'], allow_pickle=True)

        # Convert to mpii joint annotation format
        for row in range(len(data)):
            data[row][1] = surreal_utils.surreal_to_mpii(data[row][1].T)

        self.train_list = data
        self.valid_list = []
        # self.train_list = data[np.where(data[:, 2] == 1)]
        # self.valid_list = data[np.where(data[:, 2] == 0)]

        self.mean, self.std = self._compute_mean()

        # Get image center point (surreal res should be 320x240)
        self.c = tuple(np.array(cv2.imread(self.train_list[0][0]).shape[:2]) / 2)

        # Setup custom domain adaptation for gaussian blur or white noise
        self.apply_gaussian_blur = kwargs['gaussian_blur']
        self.apply_white_noise = kwargs['white_noise']

        if self.apply_gaussian_blur:
            self.gaussian_kernel = noise.get_gaussian_kernel()

    def _compute_mean(self):
        meanstd_file = './data/surreal/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print('Computing mean')
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in tqdm(self.train_list):
                img_path = index[0]
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
            print('Mean computed')

        return meanstd['mean'], meanstd['std']

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor

        if self.is_train:
            sample = self.train_list[index]
        else:
            sample = self.valid_list[index]

        img_path, joints = sample

        pts = torch.Tensor(joints)
        pts = torch.cat((pts, torch.ones((16, 1))), dim=1)
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(self.c)
        s = 1

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='scanava')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Apply noise if desired
        if self.apply_gaussian_blur:
            inp = noise.gaussian_blur(inp, self.gaussian_kernel)
        elif self.apply_white_noise:
            white_noise = torch.randn(inp.shape) * 0.3
            inp += white_noise
            inp = torch.clamp(inp, 0, 1)  # Verify no color value > 1

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
        'pts': pts, 'tpts': tpts, 'target_weight': target_weight}

        return inp, target, meta


def surreal(**kwargs):
    return Surreal(**kwargs)

surreal.njoints = 16  # ugly but works
