"""Evahan的核心对象和数据结构定义"""

from pathlib import Path
from typing import NamedTuple


class EvahanOcrItem(NamedTuple):
    image_path: Path  # 图片路径
    text: str  # 图片中的文本内容

    @property
    def relative_image_path(self) -> str:
        return f"{self.image_path.parent.name}/{self.image_path.name}"

    def to_dict(self) -> dict[str, str]:
        return {
            "image_path": self.relative_image_path,
            "text": self.text,
        }


# 地域类型转换为标准Python字典时的类型定义
REGION_DICT_TYPE = dict[str, str | list[tuple[int, int]]]


class EvahanRegion(NamedTuple):
    label: str  # 版面元素类别
    text: str  # 元素中的文本内容
    points: list[
        tuple[int, int]
    ]  # 元素的顶点坐标列表，顺序：左上、右上、右下、左下

    def to_dict(self) -> REGION_DICT_TYPE:
        return {
            "label": self.label,
            "text": self.text,
            "points": self.points,
        }


class EvahanLayoutItem(NamedTuple):
    image_path: Path  # 图片路径
    regions: list[EvahanRegion]  # 版面元素列表

    @property
    def relative_image_path(self) -> str:
        return f"{self.image_path.parent.name}/{self.image_path.name}"

    def to_dict(self) -> dict[str, str | list[REGION_DICT_TYPE]]:
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


class MyException(Exception):
    """Base exception class."""

    pass
