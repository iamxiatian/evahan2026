# awq: 使用`alpaca-zh alpaca-en sharegpt-gpt4:default`作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --model /data/app/workspace/models/Qwen2.5-VL-7B-Instruct \
    --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4:default --quant_method awq