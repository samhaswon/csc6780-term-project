#!/bin/bash

#SBATCH --account=csc6780-2025f-inference
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --job-name=single-node-inference
#SBATCH --output=runs/single-node-inference-%j.out
#SBATCH --time=01:00:00

RUN_ROOT=/work/projects/csc6780-2025f-inference/sahoward42/csc6780-term-project

# What GPU type are we on?
nvidia-smi --query-gpu=name,index,driver_version,memory.total --format=csv

cd $RUN_ROOT || exit 1
spack load py-virtualenv
. "${RUN_ROOT}/venv/bin/activate"

python3 -u a100_test.py