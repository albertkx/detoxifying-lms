#dialects=('nontoxic_aae' 'nontoxic_muse' 'toxic_aae' 'toxic_muse')
#for dialect in "${dialects[@]}"; do
#	echo ${dialect}
#	train_cmd="python3 ../PPLM/run_pplm.py \
#		-D toxicity \
#		--cond_text '<|endoftext|>' \
#		--pretrained_model gpt2-medium \
#		--class_label 0 \
#		--length 30 \
#		--gamma 1.0 \
#		--num_iterations 10 \
#		--num_samples 10000000 \
#		--stepsize 0.02 \
#		--kl_scale 0.01 \
#		--gm_scale 0.9 \
#		--sample \
#		--evaluation_file ../../data/translation_pairs/split_prompts/${dialect}.txt"
#		
#	python3 /shared/group/gpu_scheduler/reserve.py --num-gpus 1 "${train_cmd}" &> ../../logs/pplm/${dialect}.txt
#don
#dialects=('nontoxic_aae1' 'nontoxic_muse1')
#dialects=('nontoxic_aae10' 'nontoxic_muse10')
export CUDA_VISIBLE_DEVICES=2
dialects=('nontoxic_tweets1')
for file in "${dialects[@]}"; do
	train_cmd="python3 ../PPLM/run_pplm.py \
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
		--discrim_path '../../checkpoints/pplm-discrim/generic_classifier_head_epoch_10.pt' \
		--top_k 50 \
		--gm_scale 0.9 \
		--sample \
		--r 1 \
		--evaluation_file ../../data/identity_mentions/chunks/${file}.txt"
	python3 /shared/group/gpu_scheduler/reserve.py --num-gpus 1 "${train_cmd}" &> ../../logs/pplm/${file}.txt
	#--evaluation_file ../../data/translation_pairs/chunks/${file}.txt &> ../../logs/pplm/${file}.txt
done
