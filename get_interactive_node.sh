#!/usr/bin/env bash

# Adjust the container IMG and MOUNTS as appropriate for your run.
#IMG='--container-image=/lustre/fsw/portfolios/llmservice/users/pzelasko/containers/nemo-24.09-06dec24.sqsh'
IMG='--container-image=/lustre/fsw/portfolios/llmservice/users/pzelasko/containers/nemo-25.02.rc7-pytorch2.6-11mar25.sqsh'
MOUNTS='--container-mounts=/lustre/fsw/portfolios/llmservice/users/pzelasko/duplex_s2s/NeMo,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data:/data,/lustre/fsw,/lustre/fsw/portfolios/llmservice,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data,/lustre/fsw/portfolios/llmservice/users/pzelasko,/lustre/fsw/portfolios/llmservice/users/kevinhu/duplex,/lustre/fsw/portfolios/llmservice/users/kevinhu/s2s'
NGPU=8

srun --account=llmservice_nemo_speechlm --partition=interactive --job-name=ml-model.canary-interactive-debug --time=4:00:00 --gpus-per-node=$NGPU --ntasks-per-node=1 $IMG $MOUNTS --pty /bin/bash
