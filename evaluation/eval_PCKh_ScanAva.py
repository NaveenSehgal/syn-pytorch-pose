import numpy as np
import os
import argparse
import scipy.io as sio

# -----------------------------PARAMS ------------------------ #
threshold = 0.5
SC_BIAS = 0.6
# ------------------------------------------------------------ #

# Get results file
parser = argparse.ArgumentParser(description='MPII PCKh Evaluation')
parser.add_argument('-r', '--result', default='checkpoint/scanava/hg-s2/preds.mat',
                    type=str, metavar='PATH', help='path to result (default: checkpoint/mpii/hg_s2_b1/preds.mat)')
args = parser.parse_args()

# Load annotations
annotations_path = "data/scanava/scanava_labels.npy"
annotations = np.load(annotations_path, allow_pickle=True)

# Filter for test samples
test_indices = np.where(annotations[:, 2] == 0)[0]
annotations = annotations[test_indices]
ground_truth = np.dstack(annotations[:, 1])  # TODO: Verify. (16, 2, 14800)

# Load results
preds = sio.loadmat(args.result)
preds = preds['preds']  # (14800, 16, 2)
preds = np.transpose(preds, [1, 2, 0])  # (16, 2, 14800)

# Calculate error
uv_error = ground_truth - preds  # (16, 2, 14800)
uv_err = np.linalg.norm(uv_error, axis=1)  # (16, 14800)
jnt_missing = np.zeros((16, uv_err.shape[1]))
jnt_visible = 1 - jnt_missing  # should be all 1s

# Get headsizes
head_idx = 9
thorax_idx = 7
heads = ground_truth[head_idx, :, :] - ground_truth[thorax_idx, :, :]  # 2, 14800
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
