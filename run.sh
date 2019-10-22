#!/bin/bash
#SBATCH --ntasks-per-node=20
#SBATCH -N 3
#SBATCH -J QueueNet
#SBATCH -o QueueNet.out
#SBATCH -e QueueNet.err
#SBATCH --time=1:59:00


#run the application:
#OpenMP settings:


mpirun -np 60 --mca btl_tcp_if_include enp97s0f1 python ./main.py

