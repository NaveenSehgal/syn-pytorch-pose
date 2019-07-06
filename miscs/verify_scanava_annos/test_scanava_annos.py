import numpy as np
import cv2

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

''' Goal: Read in scanava_labels.npy, plot a few pictures with joint labels '''
def load_and_annotate(sample):
    path, joints, _ = sample
    img = cv2.imread(path)
    
    for joint_idx, joint in enumerate(joints):
        center = joint.astype(int)
        cv2.circle(img, tuple(center), 5, (0, 255, 0), -1)
        cv2.putText(img, mpii_joint_names.get(joint_idx), tuple(center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    return img


# Load data
labels_path = '/home/sehgal.n/syn-pytorch-pose/data/scanava/scanava_labels.npy'
labels = np.load(labels_path, allow_pickle=True)  # (N, 3)
N = len(labels)

# Choose few samples to plot 
indices = np.random.choice(range(N), 15)
labels = labels[indices]

for i, sample in enumerate(labels):
    img = load_and_annotate(sample)
    cv2.imwrite('test_{}.png'.format(i), img)

