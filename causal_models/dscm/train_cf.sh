#!/bin/bash

beta=3


exp_name="mimic_crop_beta_${beta}_soft_lr_1e4_lagrange_lr_1_damping_10"

# Set parents as race, sex, age and finding
parents='age_race_sex_finding' 
# --vae_path="../checkpoints/age_race_sex_finding/mimic_512_beta_${beta}/checkpoint.pt" \
# --vae_path="../checkpoints/age_race_sex_finding/mimic_512_beta_${beta}_z_${z_max_res}_cross_${max_cross_attn_res}/checkpoint.pt" \

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus48                              # Partition (queue)
#SBATCH --gres=gpu:1  
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

source activate tian_breast

nvidia-smi

srun python train_cf.py \
    --hps mimic256_64 \
    --exp_name=$exp_name \
    --parents_x age race sex finding\
    --lr=1e-4 \
    --lr_lagrange=0.1 \
    --damping=10 \
    --bs=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --predictor_path='../pgm/checkpoints/age_race_sex_finding/sup_aux_mimic_crop/checkpoint.pt' \
    --pgm_path='../pgm/checkpoints/age_race_sex_finding/sup_pgm_mimic_crop/checkpoint.pt' \
    --vae_path="../checkpoints/age_race_sex_finding/mimic_crop_256_64_beta_${beta}/checkpoint.pt" \
EOT