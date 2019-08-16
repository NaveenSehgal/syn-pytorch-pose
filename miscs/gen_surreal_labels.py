import numpy as np
import os 

# Sample: ['28_17_c0002_90.png', '28_17_c0002_0.png', '28_17_c0002_60.png', '28_17_c0002.npy', '28_17_c0002_30.png']
path = '/scratch/sehgal.n/datasets/train_surreal_images'

annotations = {}

for (dirpath, dirnames, filenames) in os.walk(path):
    
    data_file = os.path.basename(dirpath) + ".npy"
    first_image = os.path.basename(dirpath) + "_0.png"

    if data_file in filenames and first_image in filenames:
        data = np.load(os.path.join(dirpath, data_file))
        [filenames.remove(x) for x in filenames if not x.endswith('.png')] 
        filenames.sort()
        filenames = [os.path.join(path, os.path.join(dirpath, x)) for x in filenames]
        
        assert len(filenames) == len(data), "Some wrong with data folder {}".format(dirpath)

        # Map image path to data
        images = dict(zip(filenames, data))
        annotations.update(images) 

np.save('surreal_annotations.npy', list(annotations.items()))

