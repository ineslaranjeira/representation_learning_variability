#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-400
#SBATCH --job-name=hmm_glob
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" 4_job_array_pars.txt`
parArray=($parameters)

K=${parArray[0]}
fold=${parArray[1]}
iter=${parArray[2]}

python --version
python 4_run_inference_global_fit.py "$K" "$fold" "$iter"