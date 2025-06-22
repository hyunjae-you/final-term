#!/bin/sh
#SBATCH -J null               # Job Name
#SBATCH -p gpu_edu             # Partition
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 16                  # Number of CPU cores
#SBATCH -o %x.o%j              # Output log file
#SBATCH -e %x.e%j              # Error log file
#SBATCH --time 48:00:00        # Max wall time
#SBATCH --exclusive=user       # Exclusive node access
#SBATCH --gres=gpu:2           # Request 2 GPUs

# Move to working directory
curr=`pwd`
cd $curr

module load conda/tensorflow-gpu

# Run the script with the absolute Python path from your environment
~/miniconda3/envs/cgcnn/bin/python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 ./HexOx_cifs/

