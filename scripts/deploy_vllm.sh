CUDA_VISIBLE_DEVICES=4 \
swift deploy \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --adapters output_layout/v0-20260208-160448/checkpoint-100 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_max_num_seqs 1 \
    --vllm_tensor_parallel_size 1 \
    --max_new_tokens 8192 \
    --port 8000 \
    --served_model_name Qwen2.5-VL-7B-Instruct-LoRA