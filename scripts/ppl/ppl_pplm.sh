FILE=$1
CHECKPOINT=$2
LOG_FILE=$3
python3 PPLM/run_pplm.py \
	-D toxicity \
	--pretrained_model gpt2-medium \
	--cond_text '<|endoftext|>' \
	--class_label 0 \
	--length 30 \
	--gamma 1.0 \
	--num_iterations 10 \
	--num_samples 10000000 \
	--stepsize 0.02 \
	--kl_scale 0.01 \
	--discrim_path ${CHECKPOINT} \
	--top_k 50 \
	--gm_scale 0.9 \
	--sample \
	--r 1 \
	--evaluation_file ${FILE} 2>&1 | tee ${LOG_FILE}
