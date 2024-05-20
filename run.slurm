#! /bin/sh

#SBATCH --job-name=streamvc
#SBATCH --output=/home/yandex/APDL2324a/group_4/%j.out # redirect stdout
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=4000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --mem=16000 # CPU memory (MB)
#SBATCH --cpus-per-gpu=2 
#SBATCH --gpus=4 # GPUs in total

export HF_HOME="/home/yandex/APDL2324a/group_4/newcache"
export TORCH_HOME="/home/yandex/APDL2324a/group_4"
export XDG_CAHCE_HOME="/home/yandex/APDL2324a/group_4"

export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="2"

nvidia-smi

if [ $1 = "ce" ] ; then
accelerate launch --main_process_port $((29050 + ($SLURM_JOB_ID % 100))) ./code/train.py --run-name streamvc_$SLURM_JOB_ID --batch-size 64 --lr 2e-5 --betas 0.9 0.98 --weight-decay 1e-2 --module-to-train content-encoder --limit-num-batches 1000 --log-labels-interval 20 --log-gradient-interval 20 --accuracy-interval 50 --schedualer-gamma 0.5 --schedualer-step 200 --model-checkpoint-interval 200 --limit-batch-samples 288000
elif [ $1 = "svc" ] ; then
accelerate launch --main_process_port $((29050 + ($SLURM_JOB_ID % 100))) ./code/train.py --run-name streamvc_$SLURM_JOB_ID --batch-size 4 --lr 1e-6 --betas 0.9 0.98 --weight-decay 1e-2 --module-to-train decoder-and-speaker --content-encoder-checkpoint "./model.safetensors"
fi
