#!/bin/bash
#SBATCH -A k1422
#SBATCH --ntasks-per-node=30
#SBATCH -N 4
#SBATCH -J queuenet
#SBATCH -o logs/queuenet.out
#SBATCH -e logs/queuenet.err
#SBATCH --time=2:59:00


#run the application:
#OpenMP settings:
export OMP_NUM_THREADS=1


module load openmpi
module load miniconda
source activate /project/k1422/Softwares/anaconda3/envs/project_mpi/
mpirun --mca btl_tcp_if_include ipogif0 -np 120 python ./main_cpu.py --epochs 40 --lr 0.1

#while true; do squeue -u mag0a; sleep 100; done;
