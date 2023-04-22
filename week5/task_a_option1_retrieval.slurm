#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 4096
#SBATCH -p mhigh,mhigh
#SBATCH --gres gpu:1
#SBATCH -o logs/%x_%u_%j.ou
#SBATCH -e logs/%x_%u_%j.err
python run_retrieval.py \
    --mode image_to_text  \
    --dataset_path "$1" \
    --image_encoder resnet_18 \
    --text_encoder clip \
    --embedding_size 256 \
    --checkpoint "outputs_w5_task_a/image_to_text_resnet_18_clip_embed256_lr5e-05_wd1e-05_20230421_003046/models/epoch_15.pt"
