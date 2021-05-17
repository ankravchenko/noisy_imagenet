#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J model_accuracy
#SBATCH --output=/home/annakravchenko/geirhos/imagenet/logs/slurm-%j.out
#SBATCH --error=/home/annakravchenko/geirhos/imagenet/logs/slurm-%j.err


DIR='/data/ILSVRC2012/'
PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
PRINT=100
BATCH=100
MODEL='resnet50'
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_gauss_0_for_60_epoch60.pth.tar"
AFILE="sup_RN50_gauss_0_for_60_epoch"
GAUSS=2

${PYTHON} main_val_model_accuracy_blur.py --a ${MODEL} \
--print-freq ${PRINT} \
--batch-size ${BATCH} \
--resume ${RESUME} \
--save_accuracy_file ${AFILE} \
--gauss ${GAUSS} \
--evaluate \
${DIR}
  



