#!/bin/bash

CHECKPOINT=$1
OUT_FILE=$2
NUM_SAMPLES=$3
PROMPT_FILE=$4

if [ -n "${PROMPT_FILE}" ]; then	
	python3 FT/generate.py \
		--model_type=gpt2 \
		--model_name_or_path=${CHECKPOINT} \
		--seed 1 \
		--num_return_sequences 1 \
		--length 15 \
		--k 50 \
		--temperature 1 \
		--prompt_file ${PROMPT_FILE} \
		--batch_size 1 \
		--out_file ${OUT_FILE}
else
	python3 FT/generate.py \
		--model_type=gpt2 \
		--model_name_or_path=${CHECKPOINT} \
		--seed 1 \
		--num_return_sequences ${NUM_SAMPLES} \
		--length 30 \
		--k 50 \
		--prompt '<|endoftext|>' \
		--temperature 1 \
		--batch_size 128 \
		--out_file ${OUT_FILE}
fi
