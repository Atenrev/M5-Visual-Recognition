#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4096 # 4GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written
python run_retrieval.py \
    --mode image_to_text  \
    --dataset_path "$1" \
    --image_encoder resnet_18 \
    --text_encoder bert \
    --embedding_size 256 \
    --checkpoint "outputs_w5_task_a/image_to_text_resnet_18_bert_embed256_lr5e-05_wd1e-05_20230421_003156/models/epoch_15.pt"

python run_retrieval.py \
    --mode image_to_text  \
    --dataset_path "$1" \
    --image_encoder resnet_18 \
    --text_encoder bert \
    --embedding_size 256 \
    --checkpoint "outputs_w5_task_a/image_to_text_resnet_18_bert_embed256_lr5e-05_wd1e-05_20230421_003156/models/epoch_15.pt"
