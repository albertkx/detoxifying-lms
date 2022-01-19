LR=2e-5

DATA_DIR=$1
CHECKPOINT_DIR=$2
PT_DIR=$3

CUDA_VISIBLE_DEVICES=3 python3 ./GeDi/train_GeDi.py \
	--overwrite-output-dir \
	--do_train \
	--logit_scale \
	--max_seq_length 128 \
	--overwrite_cache \
	--learning_rate ${LR} \
	--data_dir ${DATA_DIR} \
	--model_type gpt2 \
	--model_name_or_path ${PT_DIR} \
	--output_dir ${CHECKPOINT_DIR} \
	--num_train_epochs 10 \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 16 \
	--task_name SST-2 \
	--code_0 dirty \
	--code_1 clean \
	--gen_weight 0.4 \
	--fp16 \
	--save_steps 1000 \
	--overwrite_output_dir
