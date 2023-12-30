#!/bin/bash
prefix='correct_malform_debug'
prompt_type=19
dataset='dl19'
# gpt-3.5-turbo   gpt-4-0314
model='gpt-4-0314'
correct_malform=1
print_messages=0
log_filename=logs/${prefix}_${prompt_type}_${dataset}_${model}_${correct_malform}_${print_messages}.log
nohup python -u test.py \
    --prompt_type ${prompt_type} \
    --dataset ${dataset} \
    --model ${model} \
    --print_messages ${print_messages} \
    --correct_malform 1 \
    --debug 0 \
> ${log_filename} 2>&1 &