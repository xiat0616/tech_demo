#!/bin/bash
beta=9
z_max_res=32
exp_name="padchest224_224_beta_${beta}"
parents='scanner_sex_finding'

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpus24            # partition (queue). Either gpus, gpus24, gpus48
#SBATCH --gres=gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --output=checkpoints/$parents/$exp_name/slurm.%j.log

source activate tian_breast


srun python main.py \
    --hps padchest224_224 \
    --lr 1e-3 \
    --batch_size 16 \
    --wd 5e-2 \
    --epochs 1000 \
    --exp_name=$exp_name \
    --context_dim 3 \
    --parents scanner sex finding \
    --beta=$beta \
    --bottleneck 4 \
    --z_max_res=$z_max_res \
EOT

