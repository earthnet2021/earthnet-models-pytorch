#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=6-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
# #SBATCH --exclusive
#SBATCH --job-name en23

# $1 /Net/Groups/BGI/scratch/ppandey/earthnet-models-pytorch/configs/en23/conv3d/base/base.yaml

echo "testing conv3d model"
module load cuda proxy 

cd /Net/Groups/BGI/scratch/ppandey/earthnet-models-pytorch/
srun /Net/Groups/BGI/scratch/ppandey/miniconda3/envs/earthnet2/bin/python /Net/Groups/BGI/scratch/ppandey/earthnet-models-pytorch/scripts/test.py /Net/Groups/BGI/scratch/ppandey/earthnet-models-pytorch/configs/en23/conv_3d/first_lr_1e-3_4x4.yaml /Net/Groups/BGI/scratch/ppandey/earthnet-models-pytorch/experiments/en23/cnn3d/conv_3d/config_first_lr_1e-3_4x4/checkpoints/last.ckpt iid
echo "done!"