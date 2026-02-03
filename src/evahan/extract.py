"""解析LLM的输出结果，转变为结构化数据。"""

import ast
from pathlib import Path
from typing import cast

import structlog
from bs4 import BeautifulSoup

from evahan.dataset import EvahanLayoutItem, EvahanRegion


logger = structlog.get_logger(__name__)


def extract_layout_item(image_path: Path, llm_text: str) -> EvahanLayoutItem:
    """
        解析布局生成模型的输出，将其转换为EvaHan2026格式。
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

    layout_item = EvahanLayoutItem(
        image_path=image_path,
        regions=[],
    )

    # 1. 解析HTML内容
    soup = BeautifulSoup(llm_text, "html.parser")

    for div in soup.find_all("div"):
        # 提取class属性（如果没有则返回空字符串）
        class_value = div.get("class", [""])[0]  # class返回的是列表，取第一个值
        # 提取data-bbox属性（如果没有则返回空字符串）
        data_bbox_value = div.get("data-bbox", "")
        # 提取div中间的文本内容（去除首尾空白）
        text_content = div.get_text(strip=True)

        try:
            # 将字符串形式的列表转换为实际的列表对象
            data_bbox_value = ast.literal_eval(data_bbox_value)
            points = cast(list[tuple[int, int]], data_bbox_value)

            if class_value != "text":
                text_content = ""  # 非文字区域，文本内容为空
            layout_item.regions.append(
                EvahanRegion(
                    label=class_value, points=points, text=text_content
                )
            )
        except Exception as e:
            logger.warning(f"解析div布局项失败，忽略该元素：{div}", error=e)

    return layout_item
