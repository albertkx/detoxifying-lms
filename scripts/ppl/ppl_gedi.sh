#!/bin/bash

gedi_model_path=$1
eval_file=$2
out_file=$3

python3 GeDi/generate_GeDi.py \
	--model_type gpt2 \
	--gen_model_name_or_path gpt2-medium \
	--gedi_model_name_or_path ${gedi_model_path} \
	--gen_length 30 \
	--mode detoxify \
	--num_gen 1 \
	--prompt '<|endoftext|>' \
	--do_sample \
	--disc_weight 0.3 \
	--k 50 \
	--temperature 1 \
	--evaluation_file ${eval_file} 2>&1 | tee ${out_file}
