CUDA_VISIBLE_DEVICES=2 \
swift deploy \
    --model /mnt/public/xiatian/workspace/models/Qwen2.5-VL-7B-Instruct \
    --adapters output_layout_v3/v1-20260209-030026/checkpoint-1256 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_max_num_seqs 1 \
    --vllm_tensor_parallel_size 1 \
    --max_new_tokens 2048 \
    --port 8000 \
    --served_model_name Qwen2.5-VL-7B-Instruct-LoRA
