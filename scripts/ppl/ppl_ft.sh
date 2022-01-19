#!/bin/bash

CHECKPOINT=$1
eval_data_dir=$2
LOG_DIR=$3

python3 FT/train.py --output_dir ../trash/ \
	--model_type=gpt2 \
	--model_name_or_path=${CHECKPOINT} \
	--do_eval \
	--eval_data_file ${eval_data_dir} \
	--logging_dir ${LOG_DIR} \
	--evaluation_strategy steps \
	--eval_steps 500 \
	--line_by_line
