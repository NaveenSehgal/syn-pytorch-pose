import numpy as np

mpii_joint_names = {
    0: 'R ankle',
    1: 'R knee',
    2: 'R hip',
    3: 'L hip',
    4: 'L knee',
    5: 'L ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper neck',
    9: 'head top',
    10: 'R wrist',
    11: 'R elbow',
    12: 'R shoulder',
    13: 'L shoulder',
    14: 'L elbow',
    15: 'L wrist'
}

# Taken from https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
surreal_joint_names = [
    'hips',  # 0
    'leftUpLeg',
    'rightUpLeg',
    'spine',
    'leftLeg',  # 4
    'rightLeg',
    'spine1',
    'leftFoot',
    'rightFoot',  # 8
    'spine2',
    'leftToeBase',
    'rightToeBase',  # 11
    'neck',
    'leftShoulder',
    'rightShoulder',  # 14
    'head',
    'leftArm',  # 16
    'rightArm',
    'leftForeArm',
    'rightForeArm',  # 19
    'leftHand',
    'rightHand',
    'leftHandIndex1',  # 22
    'rightHandIndex1']

mpii_index_to_surreal = {
    # NOTE: the surreal name sides are flipped. i.e., leftFoot in surreal maps to right ankle in MPII
    0: surreal_joint_names.index('leftFoot'),
    1: surreal_joint_names.index('leftLeg'),
    2: surreal_joint_names.index('leftUpLeg'),
    3: surreal_joint_names.index('rightUpLeg'),
    4: surreal_joint_names.index('rightLeg'),
    5: surreal_joint_names.index('rightFoot'),
    6: surreal_joint_names.index('hips'),
    7: surreal_joint_names.index('neck'),
    8: surreal_joint_names.index('head'),
    10: surreal_joint_names.index('leftHand'),
    11: surreal_joint_names.index('leftForeArm'),
    12: surreal_joint_names.index('leftArm'),
    13: surreal_joint_names.index('rightArm'),
    14: surreal_joint_names.index('rightForeArm'),
    15: surreal_joint_names.index('rightHand'),
}


def surreal_to_mpii(surreal_joints):
    # Note: the sides are switched in the surreal_joint_names list for some reason
    mpii_joints = np.zeros((16, 2))

    # Map joints that have direct match
    for mpii_index, surreal_idx in mpii_index_to_surreal.items():
        mpii_joints[mpii_index] = surreal_joints[surreal_idx]

    # Interpolate for top of the head (upper neck to lower neck is 1/3 distance from lower neck to head)
    mpii_joints[9] = mpii_joints[8] + (mpii_joints[8] - mpii_joints[7]) * 3

    return mpii_joints.astype(int)
