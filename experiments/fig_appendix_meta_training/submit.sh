#!/bin/bash
# This job runs with:
# - 1 node (the '-N' option)
# - 1 CPU (the '--cpus-per-task' option)
# - 1 GPU (the '--gpus-per-node' option)
# - 50GB of RAM (the '--mem' option)
# - Maximum of 5 minutes of run time (the '-t' option)
#
# If you wanted to run with multiple GPUs, you can request up to 8 per node.
# The total number of GPUs requested is 'number-of-nodes * gpus-per-node'. For Python applications,
# it's difficult to work with multiple nodes, so you probably ought to leave -N set to 1.
#SBATCH -J cuquantum_example
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB
#SBATCH --partition=minor-use-case
#SBATCH -t 45
#SBATCH -o %u-%x-job%j.out
#SBATCH --export=ALL

# Print out Job Details
echo "Job ID: "$SLURM_JOB_ID
echo "Job Account: "$SLURM_JOB_ACCOUNT
echo "Hosts: "$SLURM_NODELIST
nvidia-smi -L
echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"
echo "------------"

# Bootstrap environment
ml cuda/12.6.2
source ~/venv/bin/activate
# run script
python train_meta.py 
echo "Job completed."