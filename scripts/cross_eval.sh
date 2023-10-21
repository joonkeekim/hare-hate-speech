TEST_DATA_DIR=data/cross_eval/hateX_test.json
BATCH_SIZE=16
EVAL_BATCH_SIZE=64
MODEL=YOUR_MODEL_HERE
WANDB_NAME=YOUR_WANDB
WANDB_ENTITY=YOUR_WANDB
WANDB_GROUP=YOUR_WANDB


python -m transfer \
	--output_dir outputs/$WANDB_GROUP/$WANDB_NAME \
	--report_to wandb \
	--wandb_name $WANDB_NAME \
	--wandb_group $WANDB_GROUP \
	--wandb_entity $WANDB_ENTITY \
	--do_eval \
	--evaluation_strategy "epoch" \
	--logging_strategy "epoch" \
    --metric_for_best_model "cls_f1" \
	--logging_steps 1 \
	--model_name_or_path $MODEL \
	--per_device_train_batch_size $BATCH_SIZE \
	--per_device_eval_batch_size $EVAL_BATCH_SIZE \
	--test_data_file $TEST_DATA_DIR \
	--remove_unused_columns=False \
	--generation_config config/generation_config.json \
	--task sbic2hate \
    --reasoning False \
	--overwrite_cache