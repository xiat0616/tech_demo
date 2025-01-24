#!/bin/bash

beta=9


exp_name="padchest_beta_${beta}_5_focus_finding_soft_lr_1e4_lagrange_lr_1_damping_10"

# Set parents as scanner, sex and age
parents='scanner_sex_finding' 
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
    --hps padchest224_224 \
    --exp_name=$exp_name \
    --parents_x scanner sex finding\
    --lr=1e-4 \
    --lr_lagrange=0.1 \
    --damping=10 \
    --bs=10 \
    --wd=0.05 \
    --eval_freq=1 \
    --predictor_path='../pgm/checkpoints/scanner_sex_finding/sup_aux_padchest/checkpoint.pt' \
    --pgm_path='../pgm/checkpoints/scanner_sex_finding/sup_pgm_padchest/checkpoint.pt' \
    --vae_path="../checkpoints/scanner_sex_finding/padchest224_224_beta_${beta}/checkpoint.pt" \
EOT
