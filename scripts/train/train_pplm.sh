#!/bin/bash
DATA_PATH=$1
MODEL_PATH=$2
PT_CHECKPOINT=$3

CUDA_VISIBLE_DEVICES=3 python3 PPLM/run_pplm_discrim_train.py \
	--dataset generic \
	--dataset_fp ${DATA_PATH}/train.tsv \
	--batch_size 64 \
	--pretrained_model ${PT_CHECKPOINT} \
	--save_model \
	--output_fp ${MODEL_PATH} \
	--epochs 20

#python3 /shared/group/gpu_scheduler/reserve.py --num-gpus 1 "${train_cmd}"
