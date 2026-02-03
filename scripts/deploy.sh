CUDA_VISIBLE_DEVICES=3 \
swift deploy \
    --model /data/app/workspace/models/Xunzi_Qwen2_VL_7B_Instruct \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --served_model_name Xunzi_Qwen2_VL_7B_Instruct
