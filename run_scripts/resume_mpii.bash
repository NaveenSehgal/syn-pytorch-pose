

# Change to root project directory
RUN_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR="$(dirname "$RUN_SCRIPTS_DIR")"
cd $PROJECT_DIR

CUDA_VISIBLE_DEVICES=0 python2 ./example/main.py \
--dataset mpii \
--arch hg \
--stack 2 \
--block 1 \
--features 256 \
--checkpoint ./checkpoint/mpii/hg-s2-b1 \
--anno-path data/mpii/mpii_annotations.json \
--image-path /scratch/sehgal.n/datasets/mpii/images
--resume ./checkpoint/mpii/hg-s2/b1/checkpoint.pth.tar

