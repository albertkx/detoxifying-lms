CHECKPOINT=$1
OUT_FILE=$2
NUM_SAMPLES=$3
PROMPT_FILE=$4

if [ -n "${PROMPT_FILE}" ]; then	
	python3 PPLM/run_pplm.py \
		-D toxicity \
		--discrim_path ${CHECKPOINT} \
		--cond_file ${PROMPT_FILE} \
		--pretrained_model gpt2-medium \
		--class_label 0 \
		--length 15 \
		--gamma 1.0 \
		--num_iterations 10 \
		--num_samples 1 \
		--stepsize 0.02 \
		--kl_scale 0.01 \
		--seed 1 \
		--top_k 50 \
		--gm_scale 0.9 \
		--sample \
		--r 1 \
		--out_file ${OUT_FILE}
else
	python3 PPLM/run_pplm.py \
		-D toxicity \
		--discrim_path ${CHECKPOINT} \
		--cond_text '<|endoftext|>' \
		--pretrained_model gpt2-medium \
		--class_label 0 \
		--length 30 \
		--gamma 1.0 \
		--num_iterations 10 \
		--num_samples ${NUM_SAMPLES} \
		--stepsize 0.02 \
		--kl_scale 0.01 \
		--seed 1 \
		--top_k 50 \
		--gm_scale 0.9 \
		--sample \
		--r 1 \
		--out_file ${OUT_FILE}
fi
