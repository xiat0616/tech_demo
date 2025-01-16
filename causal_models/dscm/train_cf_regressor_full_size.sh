#!/bin/bash

beta=3
exp_name="mimic_224_224_regressor_rand_train_batch_beta_${beta}_soft_lr_1e4_lagrange_lr_1_damping_10"

# Set parents as Left-Lung_volume, Right-Lung_volume, and Heart_volume
parents='Left-Lung_volume_Right-Lung_volume_Heart_volume'

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus48   # Partition (queue), gpus, gpus24, gpus48
#SBATCH --gres=gpu:1  
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

source activate tian_breast

nvidia-smi

srun python train_cf_regressor.py \
    --hps mimic224_224_segmentation \
    --exp_name=$exp_name \
    --parents_x Left-Lung_volume Right-Lung_volume Heart_volume \
    --lr=1e-4 \
    --lr_lagrange=0.1 \
    --damping=10 \
    --bs=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --predictor_path='../pgm/checkpoints/Left-Lung_volume_Right-Lung_volume_Heart_volume/sup_aux_mimic_224_224_regressor/checkpoint.pt' \
    --pgm_path='../pgm/checkpoints/Left-Lung_volume_Right-Lung_volume_Heart_volume/sup_pgm_mimic_224_224_regressor/checkpoint.pt' \
    --vae_path="../checkpoints/Left-Lung_volume_Right-Lung_volume_Heart_volume/mimic_crop_224_224_beta_${beta}_segmentations/checkpoint.pt" \
EOT
