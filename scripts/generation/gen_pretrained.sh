#!/bin/bash

OUT_FILE=$1
NUM_SAMPLES=$2
PROMPT_FILE=$3

if [ -n "${PROMPT_FILE}" ]; then
	python3 FT/run_generation.py \
		--model_type=gpt2 \
		--model_name_or_path=gpt2-medium \
		--seed 1 \
		--num_return_sequences 1 \
		--length 30 \
		--r 10 \
		--k 50 \
		--temperature 1 \
		--prompt_file ${PROMPT_FILE} \
		--batch_size 1 \
		--out_file ${OUT_FILE}
else
	python3 FT/run_generation.py \
		--model_type=gpt2 \
		--model_name_or_path=gpt2-medium \
		--seed 1 \
		--num_return_sequences ${NUM_SAMPLES} \
		--length 30 \
		--k 50 \
		--temperature 1 \
		--prompt '<|endoftext|>' \
		--batch_size 128 \
		--out_file ${OUT_FILE}
else
