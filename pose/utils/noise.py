""" This file contains the utils for our domain adapation / noise additions """
import torch
import math
import torch.nn as nn


def apply_white_noise(img, noise_std=0.3):
    noise = torch.randn(img.shape) * noise_std
    img += noise
    return torch.clamp(img, 0, 1)  # Verify no color value > 1


def get_gaussian_kernel(sigma=1, channels=3):
    """
    Get kernel for Gaussian blur.

    # Taken from https://discuss.pytorch.org
        /t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    """
    
    kernel_size = int(math.ceil(3 * sigma) * 2 + 1)

    # Create x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2
    variance = sigma ** 2.
    padding = kernel_size // 2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def gaussian_blur(img, gaussian_kernel):
    img = img.unsqueeze(0)
    img = gaussian_kernel.forward(img)
    img = img.squeeze(0)
    return img

