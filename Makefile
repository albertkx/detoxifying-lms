data:
	mkdir -p data/eval/translation_pairs/scored/ data/eval/translation_pairs/filtered/ data/train/pplm data/train/gedi data/train/ft data/pt
	python3 scripts/data-processing/make_train_data.py \
		--path data/raw/civilcomments/train.csv \
		--ft-output data/train/ft \
		--gedi-output data/train/gedi \
		--pplm-output data/train/pplm 
	python3 scripts/score_generations.py \
		data/raw/translation_pairs/aave_samples.txt \
		data/eval/translation_pairs/scored/aave_samples_scores.jsonl
	python3 scripts/score_generations.py \
		data/raw/translation_pairs/wae_samples.txt \
		data/eval/translation_pairs/scored/wae_samples_scores.jsonl
	python3 scripts/data-processing/make_eval_data.py \
		data/eval/translation_pairs/scored \
		data/eval/translation_pairs/filtered
	head -n 1000000 data/raw/twitter/data.txt > data/train/pt/train.tsv
	tail -n +1000001 data/raw/twitter/data.txt > data/train/pt/valid.tsv

train-pt:
	./scripts/train/pretrain.sh \
		data/pt \
		checkpoints/pt \

train-ft:
	./scripts/train/finetune.sh \
		data/train/ft \
		checkpoints/ft \
		checkpoints/pt/checkpoint-10000

train-pplm:
	./scripts/train/train_pplm.sh \
		data/train/pplm \
		checkpoints/pplm \
		checkpoints/pt/checkpoint-10000

train-gedi:
	./scripts/train/train_gedi.sh \
		data/train/gedi \
		checkpoints/gedi \
		checkpoints/pt/checkpoint-10000

eval-pplm:
	./scripts/ppl/ppl_pplm.sh \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt \
		checkpoints/pplm/generic_classifier_head_epoch_10.pt \
		logs/pplm/nontoxic_wae.txt

eval-gedi:
	./scripts/ppl/ppl_gedi.sh \
		checkpoints/gedi \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt \
		logs/gedi/nontoxic_wae.txt 

eval-pt:
	./scripts/ppl/ppl_ft.sh \
		checkpoints/pt/checkpoint-10000 \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt \
		logs/pt/nontoxic_wae.txt

eval-ft:
	./scripts/ppl/ppl_ft.sh \
		checkpoints/ft/ \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt \
		logs/ft/nontoxic_wae.txt

gen-gedi-unprompted:
	./scripts/generation/gen_gedi.sh \
		checkpoints/gedi/ \
		generations/gedi/nontoxic_wae.txt \
		10 \
		gpt2-medium 
gen-gedi-prompted:
	./scripts/generation/gen_gedi.sh \
		checkpoints/gedi/ \
		generations/gedi/nontoxic_wae_prompted.txt \
		10 \
		gpt2-medium \
		data/prompts/splits/nontoxic_wae.txt

gen-pplm-unprompted:
	./scripts/generation/gen_pplm.sh \
		checkpoints/pplm/generic_classifier_head_epoch_10.pt \
		generations/pplm/nontoxic_wae.txt \
		10

gen-pplm-prompted:
	./scripts/generation/gen_pplm.sh \
		checkpoints/pplm/generic_classifier_head_epoch_10.pt \
		generations/pplm/nontoxic_wae_prompted.txt \
		10 \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt 

gen-ft-unprompted:
	./scripts/generation/gen_finetuned.sh \
		checkpoints/ft/ \
		generations/ft/nontoxic_wae.txt \
		10

gen-ft-prompted:
	./scripts/generation/gen_finetuned.sh \
		checkpoints/ft/ \
		generations/ft/nontoxic_wae_prompted.txt \
		10 \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt

gen-pt-unprompted:
	./scripts/generation/gen_finetuned.sh \
		checkpoints/pt/checkpoint-10000 \
		generations/pt/nontoxic_wae.txt \
		10

gen-pt-prompted:
	./scripts/generation/gen_finetuned.sh \
		checkpoints/pt/checkpoint-10000 \
		generations/pt/nontoxic_wae_prompted.txt \
		10 \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt

gen-resample-unprompted:
	./scripts/generation/gen_resample.sh \
		checkpoints/pplm/generic_classifier_head_epoch_10.pt \
		generations/resample/nontoxic_wae.txt \
		10

gen-resample-prompted:
	./scripts/generation/gen_resample.sh \
		checkpoints/pplm/generic_classifier_head_epoch_10.pt \
		generations/resample/nontoxic_wae_prompted.txt \
		10 \
		data/eval/translation_pairs/filtered/nontoxic_wae.txt

generations/pplm/%.json: generations/pplm/%.txt
	python3 scripts/score_generations.py $< $@
	
.PHONY:	data
