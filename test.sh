#!/bin/bash
prefix='test'
prompt_type=4
dataset='dl19'
# gpt-3.5-turbo   gpt-4-0314
model='gpt-3.5-turbo'
log_filename=logs/cot_prompt_${prefix}_${prompt_type}_${dataset}_${model}.log
nohup python -u test.py \
    --prompt_type ${prompt_type} \
    --dataset ${dataset} \
    --model ${model} \
> ${log_filename} 2>&1 &