from plot_utils import *
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

CHECKPOINT_DIR = "/home/sehgal.n/syn-pytorch-pose/checkpoint"
DATASETS = ['mpii', 'scanava', 'ac2d', 'sjl']
DATASETS = ['ac2d']
model_info = {
    'ScanAva2': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000"), None],
    'ScanAva2_gblur': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-gblur"), "gauss"],
    'ScanAva2_wnoise': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-wnoise"), "white"],
    'ScanAva2_cycle': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-cycle"), None],
    'ScanAva2_cycle_bg': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-cycle-bg"), None],
    "MPII": [os.path.join(CHECKPOINT_DIR, "mpii/mpii-hg-s2-b1-8000"), None],
}

if __name__ == '__main__':
    for dataset in DATASETS:
        fig, ax = plt.subplots()
        pcks = {}
        
        print("Running the following models on {}:\n{}".format(dataset, model_info.keys()))
        for model_name in model_info:
            path, noise = model_info[model_name]
            model = ModelMetrics(path, label=model_name)
            pck = model._calculate_pckh(dataset, noise=noise)
            
            x, y = pck[0], pck[1]
            y = np.mean(y, axis=1)
            ax.plot(x, y, label=model_name)
            pcks[model_name] = pck
            print('Finish model {} | dataset {}'.format(model_name, dataset))

        ax.legend()
        ax.set_xlabel('Normalized Distance')
        ax.set_ylabel('Detection Rate, %')
        np.save('{}.npy'.format(dataset), pcks)
        plt.savefig('{}.png'.format(dataset), bbox_inches='tight')
        print(" *** Finished dataset {} *** ".format(dataset))

