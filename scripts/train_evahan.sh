CUDA_VISIBLE_DEVICES=0,1 \
MAX_PIXELS=409600 \
swift sft \
    --model "./downloads/Qwen2.5-VL-7B-Instruct" \
    --dataset dataset/EvaHan/train_data/Swift_A.jsonl dataset/EvaHan/train_data/Swift_B.jsonl dataset/EvaHan/train_data/Swift_C.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --max_pixels=409600 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_llm true \
    --freeze_aligner false \
    --freeze_vit false \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4