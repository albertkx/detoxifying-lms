#!/bin/bash

CHECKPOINT=$1
OUT_FILE=$2
NUM_GEN=$3
PT_CHECKPOINT=$4
PROMPT_FILE=$5

if [ -n "${PROMPT_FILE}" ]; then
	python3 GeDi/generate_GeDi.py \
		--model_type gpt2 \
		--gen_model_name_or_path ${PT_CHECKPOINT} \
		--gedi_model_name_or_path ${CHECKPOINT} \
		--gen_length 30 \
		--mode detoxify \
		--prompt_file ${PROMPT_FILE} \
		--repetition_penalty 1.0 \
		--num_gen 1 \
		--out_file ${OUT_FILE} \
		--do_sample \
		--disc_weight 0.5 \
		--k 50 \
		--temperature 1
else
	python3 GeDi/generate_GeDi.py \
		--model_type gpt2 \
		--gen_model_name_or_path gpt2-medium \
		--gedi_model_name_or_path ${CHECKPOINT} \
		--gen_length 30 \
		--mode detoxify \
		--prompt '<|endoftext|>' \
		--repetition_penalty 1.0 \
		--num_gen ${NUM_GEN} \
		--out_file ${OUT_FILE} \
		--do_sample \
		--disc_weight 0.5 \
		--k 50 \
		--temperature 1
fi
