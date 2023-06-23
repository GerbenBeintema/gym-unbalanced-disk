#!/bin/bash

# Set job requirements
#SBATCH --output=GP_NARX_vary_nanb_last_row.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=d.lehman@student.tue.nl

# activate conda environment
source activate ml4sc

# run your code
python GP_NARX_vary_nanb.py