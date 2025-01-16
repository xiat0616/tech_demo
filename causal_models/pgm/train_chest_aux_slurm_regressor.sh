#!/bin/bash

exp_name="sup_aux_mimic_224_224_regressor"
# exp_name="sup_aux_mimic_224_224_with_segmentation"
parents='Left-Lung_volume_Right-Lung_volume_Heart_volume'

mkdir -p "checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1                # gpu:n, where n = number of GPUs                                       # Number of compute nodes
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log    # Output and error log

nvidia-smi

source activate tian_breast

srun python train_pgm_regressor.py \
    --exp_name=$exp_name \
    --hps mimic224_224_with_seg \
    --setup='sup_aux' \
    --parents_x Left-Lung_volume Right-Lung_volume Heart_volume \
    --lr=0.001 \
    --batch_size=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --input_res=512 \
    --enc_net="res" \ 
EOT