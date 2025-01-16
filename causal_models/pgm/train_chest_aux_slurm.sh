#!/bin/bash

exp_name="sup_aux_mimic_crop"
parents="age_race_sex_finding"
mkdir -p "checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH -p gpus                                                 # Partition (queue)
#SBATCH --nodes=1                                               # Number of compute nodes
#SBATCH --gres=gpu:teslap40:1                                   # Memory pool for all cores
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log    # Output and error log

nvidia-smi

source activate tian_breast

srun python train_pgm.py \
    --exp_name=$exp_name \
    --hps mimic256_64 \
    --setup='sup_aux' \
    --parents_x age race sex finding \
    --lr=0.001 \
    --batch_size=32 \
    --wd=0.05 \
    --eval_freq=1 \
    --input_res=512 \
    --enc_net="cnn" \ 
EOT