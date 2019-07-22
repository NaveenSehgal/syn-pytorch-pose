import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
import subprocess
from eval_PCKh import main as eval_mpii
from eval_PCKh_ScanAva import eval_scanava
from eval_pckh_ac2d import eval_ac2d
sns.set_style("whitegrid")

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
annotation_paths = {
    'scanava': os.path.join(project_dir, "data/scanava/scanava_labels.npy"),
    'mpii': os.path.join(project_dir, "data/mpii/mpii_annotations.json"),
    "ac2d": os.path.join(project_dir, "data/AC2d/ac2d_00_annotations.npy"),
    "sjl": os.path.join(project_dir, 'data/AC2d/ac2d_sjl_00_annotations.npy'),
}

image_paths = {
    'scanava': os.path.join(project_dir, "/scratch/sehgal.n/datasets/synthetic"),
    'mpii': os.path.join(project_dir, "/scratch/sehgal.n/datasets/mpii/images"),
    "ac2d": os.path.join(project_dir, "data/AC2d/ac2d_00/images"),
    "sjl": os.path.join(project_dir, 'data/AC2d/ac2d_sjl_00/images'),
}


class ModelMetrics:
    def __init__(self, folder_path, label=None):
        self.folder_path, self.train_acc, self.val_acc = folder_path, None, None
        self.exp_name = os.path.basename(os.path.normpath(folder_path))
        self.load_log()
        self.label = [label if label else self.exp_name][0]

    def load_log(self):
        """ Load log.txt file from model folder for train and validation accuracy """
        df = pd.read_csv(os.path.join(self.folder_path, "log.txt"), sep='\t')
        self.train_acc = df['Train Acc'].values
        self.val_acc = df['Val Acc'].values

    def plot_train_val(self):
        """ Plot train and val accuracy vs epochs for this model """
        fig, ax = plt.subplots()
        epochs = np.arange(1, len(self.train_acc) + 1)

        ax.plot(epochs, self.train_acc, label="Train")
        ax.plot(epochs, self.val_acc, label="Validation")
        ax.set_title(self.label)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epochs")

        return fig

    def add_train(self, ax):
        """ Add train accuracy from this model to a provided axes object """
        epochs = np.arange(1, len(self.train_acc) + 1)
        ax.plot(epochs, self.train_acc, label=self.label)

    def _calculate_pckh(self, dataset, noise):
        """ Run evaluation of this model for the given dataset """
        self._run_eval(dataset, noise)
        pred_file = os.path.join(self.folder_path, "preds_valid.mat")
        anno_path = annotation_paths.get(dataset)

        if dataset == 'mpii':
            class Args:
                def __init__(self, result):
                    self.result = result

            args = Args(pred_file)
            pck = eval_mpii(args)

        elif dataset == 'scanava':
            pck = eval_scanava(pred_file, anno_path)

        elif dataset == 'ac2d' or dataset == 'sjl':
            pck = eval_ac2d(pred_file, anno_path)

        return pck

    def _run_eval(self, dataset, noise):
        """ Run this model in evaluation mode over the dataset to generate a preds.mat file for calculating PCK """
        if dataset not in annotation_paths.keys():
            raise ValueError("Dataset {} is not a valid choice. "
                             "Please choose from ['mpii', 'scanava', 'ac2d', 'sjl']".format(dataset))

        noise_cmd = ""
        if noise == "gauss":
            noise_cmd = " --gaussian-blur "
        elif noise == "white":
            noise_cmd = " --white-noise "

        command = \
            "python2 {} --dataset {} --arch hg --stack 2 --block 1" \
            " --features 256 --anno-path {} --image-path {} --resume {} --checkpoint {} -e {}".format(
                os.path.join(project_dir, "example/main.py"), dataset, annotation_paths.get(dataset),
                image_paths.get(dataset), os.path.join(self.folder_path, "checkpoint.pth.tar"), self.folder_path,
                noise_cmd)

        # Run evaluation
        print("***** Running evaluation for {} on {} with the following command:\n{}".format(
            self.label, dataset, command))
        subprocess.call(command, shell=True)

    def add_pckh(self, ax, dataset, noise):
        assert noise is None or noise == "gauss" or noise == "white", "Choose 'gauss' or 'white' for noise"

        pck = self._calculate_pckh(dataset, noise)
        x, y = pck[0], pck[1]
        ax.plot(x, y, label=self.label)


if __name__ == '__main__':
    path = os.path.join(project_dir, "checkpoint/scanava/sa-hg-s2-b1-8000")
    model = ModelMetrics(path)

    fig, ax = plt.subplots()

    model.add_pckh(ax, 'mpii', None)
    # fig = model.plot_train_val()
    # plt.show(fig)