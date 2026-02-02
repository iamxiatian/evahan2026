CUDA_VISIBLE_DEVICES=3 \
swift deploy \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --served_model_name Qwen2.5-VL-7B-Instruct
