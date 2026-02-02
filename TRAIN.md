# 20260202 训练过程说明
## 训练数据处理
1. 使用 src/evahan/dataset_bbox.py 处理  
2. 配置文件使用 src/evahan/config.py
3. 修改内容：
- LAYOUT_USER_QUERY 修改为 data-bbox="[x1, y1, x2, y2]"
- max_pixels = 640 * 640
- 在 dataset_bbox.py转换时，使用 smart_resize 获取训练时的真实输入尺寸，然后将版面坐标 rescale 到该尺寸

## 训练脚本
1. 训练脚本使用：scripts/train_evahan.sh
2. 参数说明:
- 冻结 llm，训练 Vit 和 Aligner
- 最大输出 max_length = 8192
- lora 参数使用 lora_rank=8，lora_alpha=32



