CUDA_VISIBLE_DEVICES=2 \
swift deploy \
    --model /mnt/public/xiatian/workspace/models/Qwen2.5-VL-7B-Instruct \
    --adapters output_ocr_v1/v0-20260207-192519/checkpoint-5850 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_max_num_seqs 1 \
    --vllm_tensor_parallel_size 1 \
    --max_new_tokens 2048 \
    --port 8002 \
    --served_model_name Qwen2.5-VL-7B-Instruct-LoRA
