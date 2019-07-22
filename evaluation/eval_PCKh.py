from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
from utils import visualize
import numpy as np
import os
import argparse


def main(args):
    detection = loadmat('evaluation/data/detections.mat')
    det_idxs = detection['RELEASE_img_index']  # (1, 2958) array, max val of 24985
    debug = 0
    threshold = 0.5
    SC_BIAS = 0.6

    pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

    dict = loadmat('evaluation/data/detections_our_format.mat')
    dataset_joints = dict['dataset_joints']  # (1, 16)
    # dataset_joints: rank, rkne, rhip, lhip, lkne, lank, pelv, thor, neck, head, rwri, relb, sho
    # lshow, lelb, lwri
    jnt_missing = dict['jnt_missing']  # (16, 2958) boolean array (values of 0 or 1)
    pos_pred_src = dict['pos_pred_src']  # (16, 2, 2958)
    pos_gt_src = dict['pos_gt_src']  # (16, 2, 2958)
    headboxes_src = dict['headboxes_src']  # (2, 2, 2958)



    #predictions
    predfile = args.result  # Path to preds.mat
    preds = loadmat(predfile)['preds']  # (2958, 16, 2)
    pos_pred_src = transpose(preds, [1, 2, 0])  # (16, 2, 2958)


    if debug:

        for i in range(len(det_idxs[0])):
            anno = mat['RELEASE']['annolist'][0, 0][0][det_idxs[0][i] - 1]
            fn = anno['image']['name'][0, 0][0]
            imagePath = 'data/mpii/images/' + fn
            oriImg = sio.imread(imagePath)
            pred = pos_pred_src[:, :, i]
            visualize(oriImg, pred, pa)


    head = np.where(dataset_joints == 'head')[1][0]  # 9
    lsho = np.where(dataset_joints == 'lsho')[1][0]  # 13
    lelb = np.where(dataset_joints == 'lelb')[1][0]  # 14
    lwri = np.where(dataset_joints == 'lwri')[1][0]  # 15
    lhip = np.where(dataset_joints == 'lhip')[1][0]  # 3
    lkne = np.where(dataset_joints == 'lkne')[1][0]  # 4
    lank = np.where(dataset_joints == 'lank')[1][0]  # 5

    rsho = np.where(dataset_joints == 'rsho')[1][0]  # 12
    relb = np.where(dataset_joints == 'relb')[1][0]  # 11
    rwri = np.where(dataset_joints == 'rwri')[1][0]  # 10
    rkne = np.where(dataset_joints == 'rkne')[1][0]  # 1
    rank = np.where(dataset_joints == 'rank')[1][0]  # 0
    rhip = np.where(dataset_joints == 'rhip')[1][0]  # 2

    jnt_visible = 1 - jnt_missing  # (16, 2958) boolean array - 0 or 1
    uv_error = pos_pred_src - pos_gt_src  # (16, 2, 2958)
    uv_err = np.linalg.norm(uv_error, axis=1)  # (16, 2958)

    # np.mean(np.linalg.norm((pos_gt_src[head, :, :] - pos_gt_src[thorax, :, :]), axis=0))  127.54382142077498
    # np.mean(np.linalg.norm(headboxes_src[1, :, :] - headboxes_src[0, :, :], axis=0)) 134.10911405729254

    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]  # (2, 2958)
    headsizes = np.linalg.norm(headsizes, axis=0)  # (2958,)
    headsizes *= SC_BIAS

    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))  # (16, 2958)
    scaled_uv_err = np.divide(uv_err, scale)  # (16, 2958)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)  # (16, 2958)
    jnt_count = np.sum(jnt_visible, axis=1)  # (16,)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)  # (16, 2958) 0 or 1 bool array
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)  # (16,)


    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    name = predfile.split(os.sep)[-1]
    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    print('\nPrediction file: {}\n'.format(args.result))
    print("Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean")
    print('{:.2f}  {:.2f}     {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
            , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
            , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))

    return [threshold, pckAll]
    # print(pckAll)
    # print("Shape: {}".format(pckAll.shape))
    # np.save('pck.npy', pckAll)
    # print('Saved pck!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPII PCKh Evaluation')
    parser.add_argument('-r', '--result', default='checkpoint/mpii/hg_s2_b1/preds.mat',
                        type=str, metavar='PATH',
                        help='path to result (default: checkpoint/mpii/hg_s2_b1/preds.mat)')
    args = parser.parse_args()
    main(args)
