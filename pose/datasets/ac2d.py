from __future__ import print_function, absolute_import

import os
import numpy as np
import glob
import cv2

import torch
import torch.utils.data as data
import pose.utils.noise as noise

from pose.utils.imutils import *
from pose.utils.osutils import *
from pose.utils.transforms import *


class AC2d(data.Dataset):
    def __init__(self, is_train=False, **kwargs):
        self.img_folder = kwargs['image_path']
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']

        # Get image paths
        image_paths = sorted(glob.glob(os.path.join(self.img_folder, "*.jpg")))

        # Get annotations
        annotations = np.load(kwargs['anno_path'], allow_pickle=True)
        joints = annotations[0]  # (N, 16, 2)

        data = []
        for i in range(len(joints)):
            data.append([image_paths[i], joints[i], annotations[1][0][i]])
        data = np.array(data)  # (N, 3)

        self.train_list = self.valid_list = data

        # Get image center points
        self.c = []
        for (img_path, _, _) in data:
            img = cv2.imread(img_path)
            center = tuple(np.array(img.shape[:2]) // 2)
            self.c.append(center)

        # Setup custom domain adaptation for gaussian blur or white noise
        self.apply_gaussian_blur = kwargs['gaussian_blur']
        self.apply_white_noise = kwargs['white_noise']

        if self.apply_gaussian_blur:
            self.gaussian_kernel = noise.get_gaussian_kernel()

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor

        sample = self.valid_list[index]
        path, joints, _ = sample

        pts = torch.Tensor(joints)
        pts = torch.cat((pts, torch.ones((16, 1))), dim=1)

        c = torch.Tensor(self.c[index])
        s = max(np.array(c) * 2 // self.inp_res)

        # Adjust center / scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # Single-person pose with a centered/scaled figure
        nparts = pts.size(0)
        inp, orig_shape = resize_and_load(path)

        # Resize pts
        pts[:, 1] /= float(orig_shape[0]) / self.inp_res
        pts[:, 0] /= float(orig_shape[1]) / self.inp_res

        r = 0

        # Prepare image and groundtruth path
        # inp = crop(inp, c, s, [self.inp_res, self.inp_res], rot=r)

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
            if tpts[i, 1] > 0:
                scale = float(self.out_res) / self.inp_res
                tpts[i, 0:2] *= scale
                # tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, 1/200, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta
        meta = {"index": index, 'center': c, 'scale': s, 'pts': pts, 'tpts': tpts, 'target_weight': target_weight}
        return inp, target, meta


def ac2d(**kwargs):
    return AC2d(**kwargs)

ac2d.njoints = 16
