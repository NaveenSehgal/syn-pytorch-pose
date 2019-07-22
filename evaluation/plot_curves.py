from plot_utils import *
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

CHECKPOINT_DIR = "/home/sehgal.n/syn-pytorch-pose/checkpoint"
model_info = {
    'ScanAva2': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000"), None]
    'ScanAva2_gblur': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-gblur"), "gauss"]
    'ScanAva2_wnoise': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-wnoise"), "white"]
    'ScanAva2_cycle': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-cycle"), None]
    'ScanAva2_cycle_bg': [os.path.join(CHECKPOINT_DIR, "scanava/sa-hg-s2-b1-8000-cycle-bg"), None]
    "MPII": [os.path.join(CHECKPOINT_DIR, "mpii/mpii-hg-s2-b1-8000"), None]
}

if __name__ == '__main__':
    for dataset in ['mpii', 'scanava', 'ac2d', 'sjl']:
        fig, ax = plt.subplots()


        for model in model_info:
            model = ModelMetrics(model_info[model])
            model.add_pckh(ax, dataset)

        ax.legend()

        plt.savefig(fig, '{}.png'.format(dataset))
