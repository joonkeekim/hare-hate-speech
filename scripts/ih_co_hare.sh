TRAIN_DATA_DIR=data/implicit-hate/IH_exp_co_hare.json
EVAL_DATA_DIR=data/implicit-hate/IH_exp_val.json
TEST_DATA_DIR=data/implicit-hate/IH_exp_test.json

EPOCH=10
BATCH_SIZE=8
EVAL_BATCH_SIZE=32
GRAD_ACCUM_STEPS=4
MODEL=YOUR_MODEL_HERE
WANDB_NAME=YOUR_WANDB
WANDB_ENTITY=YOUR_WANDB
WANDB_GROUP=YOUR_WANDB

python -m finetune_t5 \
	--output_dir outputs/$WANDB_NAME/ \
	--report_to wandb \
	--wandb_name $WANDB_NAME \
	--wandb_group $WANDB_GROUP \
	--wandb_entity $WANDB_ENTITY \
	--do_train \
	--do_eval \
	--evaluation_strategy "epoch" \
	--logging_strategy "epoch" \
	--save_strategy "epoch" \
	--load_best_model_at_end True \
    --metric_for_best_model "cls_f1" \
	--logging_steps 1 \
	--save_steps 1 \
	--num_train_epochs $EPOCH \
	--learning_rate 5e-4 \
	--lr_scheduler_type constant \
	--optim adafactor \
	--save_total_limit 1 \
	--model_name_or_path $MODEL \
	--per_device_train_batch_size $BATCH_SIZE \
	--per_device_eval_batch_size $EVAL_BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $TRAIN_DATA_DIR \
	--eval_data_file $EVAL_DATA_DIR \
	--test_data_file $TEST_DATA_DIR \
	--remove_unused_columns=False \
	--generation_config config/generation_config.json \
	--task implicit \
	--reasoning True \
	--overwrite_cache
