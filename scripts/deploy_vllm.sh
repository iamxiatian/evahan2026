CUDA_VISIBLE_DEVICES=1 \
swift deploy \
    --model /mnt/public/lyl/Qwen3-VL/qwen-vl-finetune/output/qwen2_5-vl-7b-evahan-lora-b-augment-0209/checkpoint-3130-lora-merged \
    --model_type qwen2_5_vl \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 8192 \
    --served_model_name Qwen2.5-VL-7B-Instruct