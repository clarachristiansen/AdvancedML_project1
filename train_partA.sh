#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J "vae_flow[1-20]"
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 01:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /work3/s214659/AdvancedML/submits/outputs/train_partA_%J_%I.out
#BSUB -e /work3/s214659/AdvancedML/submits/outputs/train_partA_%J_%I.err

### Conda and cwd
cd /work3/s214659/AdvancedML || exit 1
source /work3/s214659/miniconda3/etc/profile.d/conda.sh
conda activate AdvancedML

PRIOR="flow"
SEED=$((LSB_JOBINDEX - 1))
MODEL="model_${PRIOR}_binary_seed${SEED}.pt"
OUTDIR="/work3/s214659/AdvancedML/results"
OUTFILE="${OUTDIR}/test_elbo_${PRIOR}_seed${SEED}.txt"

### Run
python3 01_vae_bernoulli.py train \
    --prior ${PRIOR} \
    --data 'binary' \
    --model "${MODEL}" \
    --seed ${SEED} \
    --epochs 10

python3 01_vae_bernoulli.py evaluate \
    --prior ${PRIOR} \
    --data 'binary' \
    --model "${MODEL}" \
    --seed ${SEED} \
    --epochs 10 \
    | grep "RESULT" >> "$OUTFILE"