# Swift Finetune

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
uv venv
uv pip install -e .
uv pip install pyav qwen_vl_utils
uv pip install torch==2.9.0
uv pip install transformers==4.57.6
uv pip install flash_attn==2.8.3
uv pip install deepspeed==0.17.6
uv pip install peft==0.18
```

## 推理测试

```bash
# Experimental environment: A100
# 30GB GPU memory
#CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-vl-7b-instruct
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-vl-7b-instruct


```

测试：

```plaintext
<<<  <image>提取图中的文字，以自然阅读顺序输出。
Input an image path or URL <<< /data/app/workspace/evahan2026/dataset/train_data/Dataset_A/a_0001.jpg
```
