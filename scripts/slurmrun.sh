#!/bin/bash
#SBATCH --partition ml
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=254000
#SBATCH --gres=gpu:6
#SBATCH --exclusive
#SBATCH --job-name earthnet
#SBATCH --mail-user vitus.benson@mailbox.tu-dresden.de
#SBATCH --mail-type ALL
#SBATCH -A p_da_earthnet

# $1 setting.yaml
# $2 train test both
# $3 test tracks list

arr=($(cat $1 | grep "Setting:"))
setting=${arr[1]}
echo $setting

echo "Sourcing bash RC"
source ~/.bashrc
echo "Activating Conda Env"
conda activate pt15

echo "Making Directories"
if [ $1 == "train" || $1 == "both"]; then
    if [ $setting == "en21-std"]; then
        cd /home/vibe622c/code/codyn-pytorch
        mkdir -p /tmp/data/release/train/
        mkdir -p /tmp/data/release/iid_test_split/
        echo "Copying Train"
        cp -rf /scratch/ws/0/vibe622c-codyn/data/release/train/. /tmp/data/release/train/
    elif [ $setting == "en21-veg"]; then
        echo "Copying Landcover"
        cp -rf /scratch/ws/0/vibe622c-codyn/data/release/landcover/. /tmp/data/release/landcover/
    else [ $setting == "europe-veg"]; then
        echo "Not Implemented"
    fi
    
    echo "Start training"
    srun python train.py configs/resnet18_enc_ndvi.yaml #codyn_lstm_big.yaml # channel_net_hr.yaml #
fi

if [ $1 == "test" || $1 == "both"]; then
    echo "Copying IID TEst"
    cp -rf /scratch/ws/0/vibe622c-codyn/data/release/iid_test_split/. /tmp/data/release/iid_test_split/
fi