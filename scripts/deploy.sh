CUDA_VISIBLE_DEVICES=4 \
swift deploy \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --adapters output_wuwen/v1-20260202-103304/checkpoint-626 \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --port 8002 \
    --served_model_name Qwen2.5-VL-7B-Instruct-LoRA
