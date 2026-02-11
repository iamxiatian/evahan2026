import os
from pathlib import Path

from dotenv import load_dotenv
from rich import print


load_dotenv()

QWEN_VL_7B_INSTRUCT = os.getenv("qwen_vl_7b_instruct")

max_pixels = int(640*640)

# 训练数据集的存放路径，该路径下存放了具体的三个数据集
EVAHAN_DATA_PATH: Path = Path(
    os.getenv("evahan_dataset_parent_path", default="./dataset")
)

# 三个数据集的路径
EVAHAN_TRAIN_PATH_A: Path = EVAHAN_DATA_PATH / "train_data/Dataset_A"
EVAHAN_TRAIN_PATH_B: Path = EVAHAN_DATA_PATH / "train_data/Dataset_B"
EVAHAN_TRAIN_PATH_C: Path = EVAHAN_DATA_PATH / "train_data/Dataset_C"
EVAHAN_TEST_PATH_B: Path = Path("dataset/TestData/Task_B")

if EVAHAN_DATA_PATH.exists():
    print(f"[green]EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH}[/green]")
else:
    print(f"[red]EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH} not exists![/red]")

# OCR的默认提取提示语
# OCR_USER_QUERY = "提取图中的文字，以自然阅读顺序输出。"
OCR_USER_QUERY = """请对这张古籍图像进行古文OCR识别，以自然阅读顺序输出。

## 特殊字符要求
- 只输出识别出的古文字，不要输出提示词
- 行内较大的空白请输出　
- 图片中存在方框形状的占位符，请输出□，不要输出口
- 缺少笔画、未知的古文字符，请输出𤣥
"""

LAYOUT_USER_QUERY = """识别并标记古籍图像中的版面元素, 输出为HTML标签格式

## 需要识别的元素类型
1. book_edge - 书籍边缘/书边
2. image - 插图、图表、绘画
3. seal - 印章区域(不识别印章文字)
4. text - 文字区域

## 要求
- 使用HTML标签格式输出，每个元素必须有data-bbox属性，其值为从bbox的x1 y1 x2 y2 四个点构成的坐标列表”
- 检测图像中的所有目标区域
- 按发现顺序输出，允许区域重叠
- 不检测book_edge元素中包含的其他元素
- 文字区域的内容对应以自然阅读顺序返回的文本内容，并滤掉空格换行等空白符号；非文字区域的内容为空。
## 示例输出：
<div class="book_edge" data-bbox="x1 y1 x2 y2"></div>
<div class="text" data-bbox="x1 y1 x2 y2">文本区域的文字内容</div>
<div class="seal" data-bbox="x1 y1 x2 y2"></div>
<div class="text" data-bbox="x1 y1 x2 y2">文本区域的文字内容</div>
<div class="image" data-bbox="x1 y1 x2 y2"></div>
"""


# LAYOUT_SYSTEM_PROMPT = """"你是古籍版面分析专家，专门检测扫描古籍中的四类元素：
# 1. book_edge - 书籍边缘/书边
# 2. image - 插图、图表、绘画
# 3. seal - 印章区域（不识别印章文字）
# 4. text - 文字区域

# ## 输出要求：
# - 检测图像中的所有目标区域
# - 按发现顺序输出，允许区域重叠
# - 每个区域返回label、points（左上角和右下角的坐标）和content（文字内容）
# - 文字区域的content字段对应以自然阅读顺序返回的文本内容，并滤掉空格换行等空白符号，其他区域为空。
# - 输出为JSON数组格式

# ## 示例输出：
# [
#     {
#         "label": "book_edge",
#         "points": [x1, y1, x2, y2],
#         "content": ""
#     },
#     {
#         "label": "text",
#         "points": [x1, y1, x2, y2],
#         "content": "这里是文字内容"
#     }
# ]
# """

# LAYOUT_USER_PROMPT = """请对这张古籍图像进行版面分析。"""
