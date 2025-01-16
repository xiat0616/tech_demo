#!/bin/bash

source activate tian_breast

beta=3
z_max_res=32

#SBATCH -c 16                       # Number of CPU Cores (16)
#SBATCH -p gpushigh                 # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 72G                  # memory pool for all cores
#SBATCH --nodelist loki          	# SLURM node
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log

exp_name="padchest224_224_beta_${beta}"
parents='scanner_age_sex_finding'
python main.py \
    --hps padchest224_224 \
    --lr 1e-3 \
    --batch_size 16 \
    --wd 5e-2 \
    --epochs 1000 \
    --exp_name=$exp_name \
    --context_dim 6 \
    --parents scanner age sex finding \
    --beta=$beta \
    --bottleneck 4 \
    --z_max_res=$z_max_res \


