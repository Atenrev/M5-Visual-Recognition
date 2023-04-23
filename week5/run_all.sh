#!/bin/bash

CUDA_VISIBLE_DEVICES=0 bash task_a_option1.slurm $1 &
CUDA_VISIBLE_DEVICES=1 bash task_a_option1_retrieval.slurm $1 &
CUDA_VISIBLE_DEVICES=2 bash task_a_option2.slurm  $1 &
CUDA_VISIBLE_DEVICES=3 bash task_a_option2_retrieval.slurm $1 &
CUDA_VISIBLE_DEVICES=4 bash task_b_option1.slurm $1 &
CUDA_VISIBLE_DEVICES=5 bash task_b_option1_retrieval.slurm $1 &
CUDA_VISIBLE_DEVICES=6 bash task_b_option2.slurm $1 &
CUDA_VISIBLE_DEVICES=7 bash task_c_option1.slurm $1 &
CUDA_VISIBLE_DEVICES=9 bash task_c_option2.slurm $1 &
echo LEVIATHAN IS UNLEASHED

