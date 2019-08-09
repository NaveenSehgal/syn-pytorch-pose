from __future__ import print_function, absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import glob
import scipy.misc
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import os

import _init_paths
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.evaluation import get_preds
import pose.models as models
import pose.losses as losses
from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
from pose.utils import noise
import numpy as np


def eval_ac2d(model_folder, use_gaussian_noise=False, use_white_noise=False):
    device = torch.device('cuda')
    print("==> creating model '{}', stacks={}, blocks={}".format('hg', 2, 1))
    njoints = 16
    model = models.__dict__['hg'](num_stacks=2, num_blocks=1, num_classes=njoints, resnet_layers=50)
    model = torch.nn.DataParallel(model).to(device)
    checkpoint_file = os.path.join(model_folder, "checkpoint.pth.tar")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])

    # Get data
    anno_path = 'data/AC2d/ac2d_00_annotations.npy'
    image_path = 'data/AC2d/ac2d_00/images'
    images = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
    annotations = np.load(anno_path, allow_pickle=True)
    joints = annotations[0]

    # Domain adaptation
    if use_gaussian_noise:
        gaussian_kernel = noise.get_gaussian_kernel()

    # Run images through model
    dataset_preds = []
    for img_path, joint in zip(images, joints):
        inp = load_image(img_path)

        # Apply noise if desired
        if use_gaussian_noise:
            inp = noise.gaussian_blur(inp, gaussian_kernel)
        elif use_white_noise:
            white_noise = torch.randn(inp.shape) * 0.3
            inp += white_noise
            inp = torch.clamp(inp, 0, 1)  # Verify no color value > 1

        # Resize to 256x256
        original_res = inp.shape[1:]
        img = im_to_torch(scipy.misc.imresize(im_to_numpy(inp), [256, 256]))
        img = img.unsqueeze(0)

        # Run through model
        results = model(img)
        output = results[-1].cpu()
        coords = get_preds(output)

        # pose-processing
        res = [64, 64]
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = output[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if px > 1 and px < res[0] and py > 1 and py < res[1]:
                    diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                    coords[n][p] += diff.sign() * .25
        coords += 0.5
        preds = coords.clone().squeeze(0).cpu().numpy()

        # Remap preds to original resolution (from 64x64 heatmap output)
        preds[:, 0] *= (original_res[1] / 64)
        preds[:, 1] *= (original_res[0] / 64)
        dataset_preds.append(preds)

    dataset_preds = np.array(dataset_preds)

    ### Calculate PCK
    ground_truth = np.transpose(joints, [1, 2, 0])
    preds = np.transpose(dataset_preds, [1, 2, 0])
    SC_BIAS = 0.6
    threshold = 0.5

    # Calculate error
    uv_error = ground_truth - preds  # (16, 2, N)
    uv_err = np.linalg.norm(uv_error, axis=1)  # (16, N)
    jnt_missing = np.zeros((16, uv_err.shape[1]))
    jnt_visible = 1 - jnt_missing  # should be all 1s

    # Get headsizes
    head_idx = 9
    thorax_idx = 7
    heads = ground_truth[head_idx, :, :] - ground_truth[thorax_idx, :, :]  # 2, N
    head_sizes = np.linalg.norm(heads, axis=0)  # (14800, )
    head_sizes *= SC_BIAS

    # Scale error for PCKh@threshold
    scale = np.multiply(head_sizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    less_than_threshold = scaled_uv_err < threshold
    jnt_count = np.sum(jnt_visible, axis=1)  # (16,)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)  # (16,)

    # Get range of PCKh values
    threshold_vals = np.arange(0, 0.5, 0.01)
    PCK_vals = np.zeros((len(threshold_vals), 16))

    for i, thresh in enumerate(threshold_vals):
        less_than_threshold = np.multiply(scaled_uv_err < thresh, jnt_visible)
        PCK_vals[i, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    return PCKh, [threshold_vals, PCK_vals]


if __name__ == '__main__':
    evaluate = {
        'test': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/old/scanava/hg-s2', False, False],
        'ScanAva2': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/scanava/sa-hg-s2-b1-8000', False, False],
        'ScanAva2_gblur': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/scanava/sa-hg-s2-b1-8000-gblur', True, False],
        'ScanAva2_wnoise': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/scanava/sa-hg-s2-b1-8000-wnoise', False, True],
        'ScanAva2_cycle': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/scanava/sa-hg-s2-b1-8000-cycle', False, False],
        'ScanAva2_cycle_bg': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/scanava/sa-hg-s2-b1-8000-cycle-bg', False, False],
        'MPII': ['/home/sehgal.n/syn-pytorch-pose/checkpoint/mpii/mpii-hg-s2-b1-8000', False, False]
    }

    fig, ax = plt.subplots()

    for model in evaluate:
        _, values = eval_ac2d(*evaluate.get(model))
        x, y = values
        y = np.mean(y, axis=1)
        ax.plot(x, y, label=model)

    ax.legend()
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection Rate, %')
    ax.set_title('Results on AC2d')
    plt.savefig('{}.png'.format('ac2d'), bbox_inches='tight')
