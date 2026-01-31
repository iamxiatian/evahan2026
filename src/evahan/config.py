import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

# 训练数据集的存放路径，该路径下存放了具体的三个数据集
EVAHAN_DATA_PATH = os.getenv("evahan_dataset_parent_path", default="./dataset")

EVAHAN_TRAIN_PATH_A = os.path.join(EVAHAN_DATA_PATH, "train_data/Dataset_A")
EVAHAN_TRAIN_PATH_B = os.path.join(EVAHAN_DATA_PATH, "train_data/Dataset_B")
EVAHAN_TRAIN_PATH_C = os.path.join(EVAHAN_DATA_PATH, "train_data/Dataset_C")

if os.path.exists(EVAHAN_DATA_PATH):
    print(f"[green]EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH}[/green]")
else:
    print(f"[red]EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH} not exists![/red]")

# OCR的默认提取提示语
OCR_USER_QUERY = "提取图中的文字，以自然阅读顺序输出。"

LAYOUT_USER_QUERY = """请对这张古籍扫描图像进行版面分析，找出所有属于以下类别的区域：book_edge（书边）、image（插图）、seal（印章）、text（文字）。注意，只需输出每个区域的类别和矩形框坐标，矩形框用左上角和右下角的坐标表示，格式为：“<类别>: [x1, y1, x2, y2]”。每个区域输出一行。"""

# LAYOUT_SYSTEM_PROMPT = """"你是一个专业的古籍版面分析专家，专门识别扫描古籍中的版面元素。你的任务是：
# 1. 识别并标注以下四类元素：book_edge(书边/边缘)、image(图像/插图)、seal(印章)、text(文字区域)
# 2. 只关注元素的位置和类别，不识别文字内容
# 3. 每个元素用矩形框标注
# 4. 允许区域重叠
# 5. 对整个图像中所有符合条件的元素进行完整标注"""

# LAYOUT_USER_PROMPT = """请对这张古籍扫描图像进行版面分析，识别出所有的book_edge、image、seal和text区域。注意：
#     1. 只标注位置，不识别文字内容
#     2. 按发现顺序输出，允许区域重叠
#     3. 每个区域返回格式：
#     {
#         "label": "类别",
#         "bbox": [x1,y1,x2,y2]
#     }
#     3.
#     4. 对于text区域，text字段留空即可"""
