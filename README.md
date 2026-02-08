# EVAHAN2026

光学字符识别（Optical Character Recognition，OCR）是一种将印刷或手写文本的图片转换为机器
编码文本的基础技术。OCR的准确率与速度直接决定着系统整体性能，并影响文档数字化、信息提取及智能
检索等下游应用的用户体验。然而，古籍文档的排版和布局与现代印刷存在显著差异，这使得基于现代文档
数据开发的OCR 技术及模型，在处理古籍相关图像时往往难以达到理想的识别效果。加之古籍本身的文字形
态复杂、版式多样等特性，古籍OCR识别至今仍是一项颇具挑战性的任务。

## 重要

dataset目录下为比赛放提供的原始数据和脚本转换后的数据，数据集较大，因此不放入git中控制，在Mac
或Linux系统下，可以通过软连接的方式链接该目录。

### 预处理

赛方提供的Dataset_A和Dataset_C两个数据集的图片阅读顺序需要旋转为正常的阅读顺序，运行以下脚本，将原始文件进行旋转并保存到新目录中。

执行脚本完成下面的预处理：

   - 解压数据集到项目根目录下的dataset目录下
   - 将数据集A和C进行原地旋转
   - 生成适合Swift框架微调的数据集，分为数组格式的json文件、一行一个对象的jsonl文件
   - 将数据集的版面标注结果进行可视化，标记元素信息的图片保存在与数据集同级的`Dataset_B_annotated`目录下

执行脚本：

```bash
uv run -m evahan.prepare /path/to/evahan2026.zip /path/to/TestSet.zip
```

此时，会在`./dataset/train_data`目录下下，生成`Swift_A.jsonl`、`Swift_B.jsonl`和`Swift_C.jsonl`，三个文件，注意，里面的图片文件采用了绝对地址。这三个文件可以直接交给Swift框架进行微调。

## 思路

### 数据增强

图片转灰度图和黑白，上下拼接，形成数据集D和E两个目录。

### 模型参数

LoRA 参数：

```plaintext
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --ddp_backend nccl\
    --num_train_epochs 8 \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --max_length 2048 \
```

### 算法

集成：通用模型、两次微调模型，投票解决冲突。


## 模型要求

- closed model: Qwen2.5-VL-7B-Instruct or Xunzi_Qwen2-VL-7B-Instruct

下载模型：

```shell
# Qwen2.5-VL-7B-Instruct直接利用modelscope下载即可
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./Qwen2.5-VL-7B-Instruct

modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --local_dir ./Qwen2.5-VL-7B-Instruct-AWQ

# Xunzi_Qwen2-VL-7B-Instruct
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
pip install aria2  -i https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com
./hdf.sh RAY5/Xunzi_Qwen2_VL_7B_Instruct --local-dir Xunzi_Qwen2_VL_7B_Instruct
```

## 测评方法

EvaHan2026数据集包括三类图像文本对：纯文本图像、混合图文图像及手写文本图像，经过自动标注及专家
修订后形成高质量的训练和测试集。数据来源包括：

- 数据集A（印刷文本）选自《四库全书》里的经史子集。
- 数据集B（混合版式）包含从《四库全书》及其他古籍中选取的混合图文数据。
- 数据集C（手写文本）涵盖手写古籍，主要为汉文佛典，包含《汉文佛典（TKH）》数据集与《汉文佛典（MTH）》数据集。

数据集被分成训练集（约15,000–30,000组）与测试集（每个子集约200–500组），所有评估数据均采用
图像-文本对形式，文本以Unicode（UTF-8）编码的txt文件存储。

### 核心评估维度

- OCR性能：准确率（Precision）、召回率（Recall）和F1值
- 生成指标：BLEU、ROUGE-1、ROUGE-2、ROUGE-L
- 版面分析指标：mAP、IoU

## 开发环境设置

1. 安装uv

   ```shell
   curl -fsSL https://get.uv.dev | bash
   # 或者通过pip安装
   pip install uv
   ```

2. 可编辑安装本项目

   ```shell
   #创建虚拟环境
   uv venv
   uv pip install -e .

   uv sync

   uv pip install flash-attn --no-build-isolation
   ```

3. 通过uv运行脚本main.py示例

   ```shell
   uv run -m evahan.main
   ```

4. 代码格式化

   ```shell
   uv run ruff check --fix src
   ```

5. 利用Swift运行测试模型

```shell
swift infer --model /path/to/your/model
```

6. git设置

```shell
git config --global core.quotepath false # 让 Git 不要将非 ASCII 字符的文件名用引号括起来
```

## 参考文献

 -[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
