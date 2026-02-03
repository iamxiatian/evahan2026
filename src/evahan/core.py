"""Evahan的核心对象和数据结构定义"""

from pathlib import Path
from typing import NamedTuple


class EvahanOcrItem(NamedTuple):
    image_path: Path  # 图片路径
    text: str  # 图片中的文本内容

    @property
    def relative_image_path(self) -> str:
        return f"{self.image_path.parent.name}/{self.image_path.name}"

    def to_dict(self) -> dict:
        return {
            "image_path": self.relative_image_path,
            "text": self.text,
        }


class EvahanRegion(NamedTuple):
    label: str  # 版面元素类别
    text: str  # 元素中的文本内容
    points: list[
        tuple[int, int]
    ]  # 元素的顶点坐标列表，顺序：左上、右上、右下、左下


class EvahanLayoutItem(NamedTuple):
    image_path: Path  # 图片路径
    regions: list[EvahanRegion]  # 版面元素列表

    @property
    def relative_image_path(self) -> str:
        return f"{self.image_path.parent.name}/{self.image_path.name}"

    def to_dict(self) -> dict:
        return {
            "image_path": self.relative_image_path,
            "regions": [
                {
                    "label": region.label,
                    "text": region.text,
                    "points": region.points,
                }
                for region in self.regions
            ],
        }
