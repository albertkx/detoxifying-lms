DATA_DIR=$1
CHECKPOINT_DIR=$2

CUDA_VISIBLE_DEVICES=3
python3 FT/train.py --output_dir=${CHECKPOINT_DIR} \
	--model_type=gpt2 \
	--model_name_or_path=gpt2-medium \
	--do_train \
	--do_eval \
	--train_data_file ${DATA_DIR}/train.tsv \
	--eval_data_file ${DATA_DIR}/valid.tsv \
	--per_device_train_batch_size 16 \
	--block_size 128 \
	--per_device_eval_batch_size 16 \
	--learning_rate 5e-5 \
	--weight_decay 0.01 \
	--adam_beta2 0.98 \
	--save_total_limit 10 \
	--save_steps 1000 \
	--fp16 \
	--warmup_steps 5000 \
	--max_grad_norm 10000000000 \
	--max_steps 10000 \
	--overwrite_output_dir \
	--evaluation_strategy steps \
	--eval_steps 500 \
	--line_by_line
