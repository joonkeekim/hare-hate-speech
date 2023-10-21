API_KEY=YOUR_KEY_HERE
# dataset
data=sbic
# dataset split name for making rationales
SPLIT=train
# false for zeroshot, true for multi-hop stage
MERGE=True
# path for request file
REQUEST_NAME=request.jsonl
# pred for zero-shot
ATTRIBUTE=pred
# set your own output file path
OUT_FILE=rationales/sbic_co_hare

python oai/format_oai_data.py \
  --data_path data/$data \
  --data $data \
  --data_split $SPLIT \
  --request_form $REQUEST_NAME \
  --model gpt-3.5-turbo-0613 \
  --max_tokens 512 \
  --temperature 0.99 \
  --num_choices 8 \
  --task_type request && \
python oai/api_request_parallel_processor.py \
  --requests_filepath $REQUEST_NAME \
  --save_filepath "${OUT_FILE}_raw.jsonl" \
  --request_url https://api.openai.com/v1/chat/completions \
  --api_key $API_KEY \
  --max_requests_per_minute 3000 \
  --max_tokens_per_minute 240000 \
  --max_attempts 10 \
  --logging_level 20 && \
python oai/format_oai_data.py \
  --data_path data/$data \
  --data $data \
  --data_split $SPLIT \
  --raw_output "${OUT_FILE}_raw.jsonl" \
  --filtered_output "${OUT_FILE}.jsonl" \
  --task_type filter \
  --attribute $ATTRIBUTE \
  --merge $MERGE
