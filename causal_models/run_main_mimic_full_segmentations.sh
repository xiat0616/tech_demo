#!/bin/bash
beta=3
z_max_res=32
exp_name="mimic_crop_224_224_beta_${beta}_segmentations"
parents='Left-Lung_volume_Right-Lung_volume_Heart_volume'



sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus48            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log

source activate tian_breast

srun python main.py \
    --hps mimic224_224_with_seg \
    --lr 1e-3 \
    --batch_size 16 \
    --wd 5e-2 \
    --epochs 1000 \
    --exp_name=$exp_name \
    --context_dim 3 \
    --parents Left-Lung_volume Right-Lung_volume Heart_volume \
    --beta=$beta \
    --bottleneck 4 \
    --z_max_res=$z_max_res \
EOT

