#!/bin/bash
prefix='debug'
prompt_type=13
dataset='dl19'
# gpt-3.5-turbo   gpt-4-0314
model='gpt-3.5-turbo'
print_messages=0
log_filename=logs/cot_prompt_${prefix}_${prompt_type}_${dataset}_${model}_${print_messages}.log
nohup python -u test.py \
    --prompt_type ${prompt_type} \
    --dataset ${dataset} \
    --model ${model} \
    --print_messages ${print_messages} \
> ${log_filename} 2>&1 &