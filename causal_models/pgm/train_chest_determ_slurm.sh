#!/bin/bash

exp_name="sup_determ_padchest"
parents="scanner_sex_finding"
mkdir -p "checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1    
#SBATCH --nodelist=monal04 # (optional, only useful on gpus partition)
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log    # Output and error log

nvidia-smi

source activate tian_breast

srun python train_pgm.py \
    --exp_name=$exp_name \
    --hps padchest224_224 \
    --setup='sup_determ' \
    --parents_x scanner sex finding \
    --lr=0.001 \
    --batch_size=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --input_res=224 \
    --enc_net="debug" \ 
EOT