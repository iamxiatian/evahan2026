"""解析LLM的输出结果，转变为结构化数据。"""

import ast
from typing import cast

import structlog
from bs4 import BeautifulSoup

from evahan.dataset import EvahanRegion


logger = structlog.get_logger(__name__)


def extract_layout_regions(llm_text: str) -> list[EvahanRegion]:
    """
        解析布局生成模型的输出，提取出其中的版面元素区域列表。
        模型输出的结果示例：

        ```html
        <div class="text" data-bbox="[[385, 507], [433, 507], [433, 690], [385, 690]]">毛邦翰補</div>
    <div class="book_edge" data-bbox="[[0, 0], [42, 0], [42, 787], [0, 787]]"></div>
    <div class="image" data-bbox="[[120, 251], [350, 251], [350, 575], [120, 575]]"></div>
        ```

        Args:
            llm_text (str): LLM的输出字符串。

        Returns:
            EvahanLayoutItem: 结构化的布局数据项。
    """

    regions: list[EvahanRegion] = []

    # 1. 解析HTML内容
    soup = BeautifulSoup(llm_text, "html.parser")

    for div in soup.find_all("div"):
        # 提取class属性（如果没有则返回空字符串）
        class_value = div.get("class")  # class返回的是列表，取第一个值
        if class_value:
            class_value = class_value[0]
        else:
            logger.warning(f"div元素缺少class属性，跳过该元素：{div}")
            continue  # 如果没有class属性，跳过该div

        # 提取data-bbox属性, 由于大模型结果的不可靠，需要滤掉开始和结尾的其他符号
        data_bbox_value: str = str(div.get("data-bbox", ""))
        p1: int = data_bbox_value.find("[")
        p2: int = data_bbox_value.rfind("]")
        data_bbox_value = data_bbox_value[p1 : p2 + 1]

        # 提取div中间的文本内容（去除首尾空白）
        text_content = div.get_text(strip=True)

        try:
            # 将字符串形式的列表转换为实际的列表对象
            data_bbox_value = ast.literal_eval(data_bbox_value)
            points = cast(list[tuple[int, int]], data_bbox_value)

            if class_value != "text":
                text_content = ""  # 非文字区域，文本内容为空
            regions.append(
                EvahanRegion(
                    label=class_value, points=points, text=text_content
                )
            )
        except Exception as e:
            logger.warning(f"解析div布局项失败，忽略该元素：{div}", error=e)

    return regions
