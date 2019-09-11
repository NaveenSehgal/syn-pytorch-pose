import numpy as np
import scipy.io as sio
import json
import argparse
import os


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


FOLDERS_TO_USE = {
    # "SYN_RR_amir_180329_0624_G20190212_1843_P2000_A00",
    # "SYN_RR_behnaz_180118_2009_G20190212_1854_P2000_A00",
    # "SYN_RR_boya_G20190216_1337_P2000_A00",
    # "SYN_RR_chen_G20190216_1438_P2000_A00",
    # "SYN_RR_dan_jacket_180521_1312_G20190605_0832_P2000_A00",
    # "SYN_RR_eddy_no_coat_180517_G20190212_1950_P2000_A00",
    # "SYN_RR_jianchao_G20190216_1507_P2000_A00",
    # "SYN_RR_jinpeng_G20190216_1929_P2000_A00",
    # "SYN_RR_kefei_G20190217_1000_P2000_A00",
    # "SYN_RR_kian_180517_1605_G20190212_1953_P2000_A00",
    # "SYN_RR_kian_jacket_180517_1617_G20190605_0832_P2000_A00",
    # "SYN_RR_naveen_180403_1612_G20190212_1953_P2000_A00",
    # "SYN_RR_naveen_180403_1635_G20190605_1348_P2000_A00",
    # "SYN_RR_ray_G20190217_1000_P2000_A00",
    # "SYN_RR_sarah_171201_1045_G20190213_0752_P2000_A00",
    # "SYN_RR_sarah_180423_1211_G20190213_0752_P2000_A00",
    # "SYN_RR_sarah_180423_1220_G20190213_0753_P2000_A00",
    # "SYN_RR_sarah_180423_1317_G20190213_0753_P2000_A00",
    # "SYN_RR_sharyu_G20190217_1238_P2000_A00",
    # "SYN_RR_shiva_G20190217_1239_P2000_A00",
    "SYN_RR_shuangjun_180403_1734_G20190213_0753_P2000_A00",
    "SYN_RR_shuangjun_180403_1748_G20190213_0810_P2000_A00",
    "SYN_RR_shuangjun_180502_1536_G20190213_0810_P2000_A00",
    "SYN_RR_shuangjun-2_G20190217_1239_P2000_A00",
    "SYN_RR_shuangjun_blackT_180522_1542_G20190604_2240_P2000_A00",
    "SYN_RR_shuangjun_blueSnow_180521_1531_G20190604_2241_P2000_A00",
    "SYN_RR_shuangjun_G20190217_1239_P2000_A00",
    # "SYN_RR_shuangjun_grayDown_180521_1516_G20190604_2241_P2000_A00",
    "SYN_RR_shuangjun_grayT_180521_1658_G20190605_1031_P2000_A00",
    # "SYN_RR_shuangjun_gridDshirt_180521_1548_G20190604_2243_P2000_A00",
    "SYN_RR_shuangjun_jacketgood_180522_1628_G20190605_0831_P2000_A00",
    "SYN_RR_shuangjun_nikeT_180522_1602_G20190605_1030_P2000_A00",
    "SYN_RR_shuangjun_whiteDshirt_180521_1600_G20190605_0834_P2000_A00",
    # "SYN_RR_steve_2_good_color_G20190605_1348_P2000_A00",
    # "SYN_RR_william_180502_1449_G20190213_0810_P2000_A00",
    # "SYN_RR_william_180502_1509_G20190213_0810_P2000_A00",
    # "SYN_RR_william_180503_1704_G20190213_0810_P2000_A00",
    # "SYN_RR_yu_170723_1000_G20190213_0810_P2000_A00",
    # "SYN_RR_zishen_G20190217_1239_P2000_A00",
}

parser = argparse.ArgumentParser()
parser.add_argument("-i,--syn-dir", type=str, dest="syn_dir", required=True, help="path to parent synthetic directory")
parser.add_argument('-o,--output', type=str, dest="output", help="output file path (should be .npy file)")
args = parser.parse_args()
syn_dir = args.syn_dir

if not os.path.exists(syn_dir):
    raise NotADirectoryError(syn_dir)

print('Loading the following ScanAva folders: ')
[print(k) for k in FOLDERS_TO_USE]

dataset = []

for folder in FOLDERS_TO_USE:
    folder_path = os.path.join(syn_dir, folder)

    # Load that folder's joint data
    X = sio.loadmat(os.path.join(folder_path, 'joints_gt.mat'))['joints_gt'][:2, :, :].swapaxes(0, 2)   # (N, 14, 2)
    mpii_joints = []

    for joint in X:
        joint_mpii = convert_syn_to_mpii(joint)
        mpii_joints.append(joint_mpii)

    mpii_joints = np.array(mpii_joints)  # (N, 16, 2)

    # Get train/test split (make last 20% test)
    N = len(X)
    num_train = int(N * 0.8)
    is_train = np.append(np.ones(num_train), np.zeros(N - num_train)).astype(int)

    # Dictionary, map each image path to ['joint data', 'is_train']
    image_paths = sorted(list(absoluteFilePaths(os.path.join(folder_path, 'images'))), key=get_image_number)

    for x, y, z in zip(image_paths, mpii_joints, is_train):
        dataset.append([x, y, z])

np.save(args.output, dataset)
