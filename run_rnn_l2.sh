#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -J rnnl2
#SBATCH -o ./logs/RNNl2.out
#SBATCH -e ./logs/RNNl2.err
#SBATCH --time=3:59:00


#run the application:
#OpenMP settings:


mpirun -n 1 --mca btl_tcp_if_include enp97s0f1 python ./main.py --file_head='RNN' --loss='l2'
