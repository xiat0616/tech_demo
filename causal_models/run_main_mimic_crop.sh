#!/bin/bash
beta=3
z_max_res=32
exp_name="mimic_crop_256_64_beta_${beta}"
parents='age_race_sex_finding'

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus24            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log

source activate tian_breast


python main.py \
    --hps mimic256_64 \
    --lr 1e-3 \
    --batch_size 16 \
    --wd 5e-2 \
    --epochs 1000 \
    --exp_name=$exp_name \
    --context_dim 6 \
    --parents age race sex finding \
    --beta=$beta \
    --bottleneck 4 \
    --z_max_res=$z_max_res \
EOT


