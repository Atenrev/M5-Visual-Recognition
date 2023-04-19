#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4096 # 4GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written
python task_a.py \
    --loss "triplet"  \
    --output_path "outputs_task_a/" \
    --dataset_path "/home/mcv/datasets/COCO/" \
	--image_encoder resnet_18 \
    --embedding_size 256 \
	--batch_size 64 \
    --epochs 20