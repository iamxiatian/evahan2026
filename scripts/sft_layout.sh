export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWIFT_PATCH_CONV3D=1
#export MASTER_PORT=29500
swift sft \
    --model /mnt/public/xiatian/workspace/models/Qwen2.5-VL-7B-Instruct \
    --dataset /mnt/public/xiatian/workspace/evahan2026/dataset/train_data/Swift_B_argument.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --ddp_backend nccl\
    --num_train_epochs 8 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --gradient_accumulation_steps 4 \
    --eval_steps -1 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output_layout_v2 \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 16
