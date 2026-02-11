export CUDA_VISIBLE_DEVICES=0
swift export \
    --adapters output/v19-20260209-112122/checkpoint-1876 \
    --merge_lora true