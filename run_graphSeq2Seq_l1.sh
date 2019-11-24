#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -J graphSeq2Seql1
#SBATCH -o ./logs/graphSeq2Seql1.out
#SBATCH -e ./logs/graphSeq2Seql1.err
#SBATCH --time=3:59:00


#run the application:
#OpenMP settings:


mpirun -n 1 --mca btl_tcp_if_include enp97s0f1 python ./main.py --file_head='graphSeq2Seq' --loss='l1'
