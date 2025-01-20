#!/bin/bash

exp_name="sup_aux_mimic_debug"
parents="sex_finding"
mkdir -p "checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1    
#SBATCH --nodelist=monal04 # (optional, only useful on gpus partition)
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log    # Output and error log

nvidia-smi

source activate tian_breast

srun python train_pgm_debug.py \
    --exp_name=$exp_name \
    --hps mimic224_224_with_seg \
    --setup='sup_aux' \
    --parents_x sex finding \
    --lr=0.001 \
    --batch_size=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --input_res=224 \
    --enc_net="debug" \ 
EOT