#!/bin/bash
#SBATCH -N 1
#SBATCH -J QueueNet
#SBATCH -o QueueNet.out
#SBATCH -e QueueNet.err
#SBATCH --time=59:00
#SBATCH --gres=gpu:gtx2080ti

#run the application:
#OpenMP settings:


python ./main_cuda.py --epochs 21 --lr 0.1
