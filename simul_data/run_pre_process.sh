#!/bin/bash
#SBATCH --ntasks-per-node=20
#SBATCH -N 3
#SBATCH -J preprocess
#SBATCH -o preprocess.out
#SBATCH -e preprocess.err
#SBATCH --time=59:00


#run the application:
#OpenMP settings:

mpirun -np 60 --mca btl_tcp_if_include enp97s0f1 python ./pre_process.py
