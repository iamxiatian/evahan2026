# 脚本参数说明

- 量化： --quant_method fp8

## 部署运行

通过`Swift deploy`我们可以部署模型，再利用OpenAI的客户端访问，方便模型运行和测试。

### 部署原始的Qwen2_VL_7B_Instruct

执行以下命令：

```shell
CUDA_VISIBLE_DEVICES=3 \
swift deploy \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --served_model_name Qwen2.5-VL-7B-Instruct
```

或者将命令保存到`deploy.sh`中，运行：

```shell
nohup ./scripts/deploy.sh > deploy_qwen.log &!
```


### 部署Xunzi_Qwen2_VL_7B_Instruct

执行以下命令：

```shell
CUDA_VISIBLE_DEVICES=3 \
swift deploy \
    --model /data/app/workspace/models/Xunzi_Qwen2_VL_7B_Instruct \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --served_model_name Xunzi_Qwen2_VL_7B_Instruct
```

或者将命令保存到`deploy.sh`中，运行：

```shell
nohup ./scripts/deploy.sh > deploy_xunzi.log &!
```


### 运行LoRA

```shell
CUDA_VISIBLE_DEVICES=3 \
nohup swift deploy \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --adapters output_wuwen/v1-20260202-103304/checkpoint-626 \
    --infer_backend pt \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --served_model_name Qwen2.5-VL-7B-Instruct-LoRA > lora_layout.log &!
```