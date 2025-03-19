#!/usr/bin/env bash

set -eou pipefail
set -x

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#NGPU=4

#export CUDA_VISIBLE_DEVICES=0
#NGPU=1

NGPU=8

export PYTHONPATH="/lustre/fsw/portfolios/llmservice/users/pzelasko/duplex_s2s/NeMo:$PYTHONPATH"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/pzelasko/hf_cache"
export WANDB_MODE=offline

#nsys profile -w true -t nvtx -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=true -x true -o duplex-tinyllama1b-profile-2.nsys-rep --force-overwrite true \
torchrun --nproc-per-node $NGPU /lustre/fsw/portfolios/llmservice/users/pzelasko/duplex_s2s/NeMo/examples/duplex_s2s/s2s_duplex_train.py \
    --config-path=/lustre/fsw/portfolios/llmservice/users/pzelasko/duplex_s2s \
    --config-name=s2s_tinyllama_repro

