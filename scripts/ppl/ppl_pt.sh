#!/bin/bash

#eval_name=$1
#eval_data_dir=../../data/twitter_eval/${eval_name}.txt
#eval_dir=../../logs/twitter_eval/${eval_name}_pt.txt
dialects=('aave_samples' 'sae_samples')
for dialect in "${dialects[@]}"; do
	train_cmd="python3 ../FT/train_finetuning.py --output_dir ../trash/ \
		--model_type=gpt2 \
		--model_name_or_path=gpt2-medium \
		--do_eval \
		--eval_data_file ./eval.txt \
		--logging_dir ../../logs/ \
		--fp16 \
		--evaluation_strategy steps \
		--eval_steps 500 \
		--line_by_line"

	#--eval_data_file ../../data/translation_pairs/src/${dialect}.txt \

	python3 /shared/group/gpu_scheduler/reserve.py --num-gpus 1 "${train_cmd}" 
	#&> ../../logs/pretrained/${dialect}.txt
done
