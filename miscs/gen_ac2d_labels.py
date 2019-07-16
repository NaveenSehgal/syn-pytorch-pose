import numpy as np
import scipy.io as sio
import json
import argparse
import os

joint_names = {
        0: "R ankle",
        1: "R knee",
        2: "R hip",
        3: "L hip",
        4: "L knee",
        5: "L ankle",
        6: "R wrist",
        7: "R elbow",
        8: "R shoulder",
        9: "L shoulder",
        10: "L elbow",
        11: "L wrist",
        12: "neck",
        13: "head",
}

def get_image_number(filename):
    # /path/to/image_0000134.png -> 134
    return int(filename.split('image_')[-1].split('.png')[0])


def convert_syn_to_mpii(synthetic_joints):
    # 14x2 scanava labels -> 16x2 mpii labels
    mpii_joints = np.zeros((16, 2))

    # Same ones
    mpii_joints[0:6] = synthetic_joints[0:6]
    mpii_joints[6] = (synthetic_joints[2] + synthetic_joints[3]) // 2
    mpii_joints[7] = synthetic_joints[12]
    mpii_joints[8] = synthetic_joints[12] + (synthetic_joints[13] - synthetic_joints[12]) / 3
    mpii_joints[9] = synthetic_joints[13]
    mpii_joints[10:] = synthetic_joints[6:12]

    return mpii_joints


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

parser = argparse.ArgumentParser()
parser.add_argument("-i,--joints-mat", type=str, dest="joints_mat", required=True, help="Mat file with joint labels. ")
parser.add_argument('-o,--output', type=str, dest="output", help="output file path (should be .npy file)")
args = parser.parse_args()

assert args.output, "You must specify an output file."

if not os.path.exists(args.joints_mat):
    raise NotADirectoryError(args.joints_mat)

X = sio.loadmat(args.joints_mat)['joints_gt']  # (3, 14, 83)
X = np.transpose(X, [2, 1, 0])  # (N, 14, 3)

# Separate 3rd channel
third_channel = X[:, :, 2]  # is this if the joint is visible?
X = X[:, :, :2]

mpii_joints = []
for joint in X:
    joint_mpii = convert_syn_to_mpii(joint)
    mpii_joints.append(joint_mpii)

mpii_joints = np.array(mpii_joints)

dataset = [mpii_joints, [third_channel, joint_names]]
np.save(args.output, dataset)
