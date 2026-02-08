"""
缩放要求：

我想对图片处理，方便QwenVL2.5更准确的定位目标的检测位置。我统一设置一个924X1232的背景图片，背景颜色为白色到灰色之间的随机背景，原始图片如果长度和宽度都小于924X1232，再将其粘贴到左上角开始的位置上，得到一张新图片；如果跟背景图片相比有一定冗余空间，则继续从左上角偏移（x，y）个位置，粘贴到背景上，并记录随机偏移值（x，y），得到另一张图片。

如果图片的长或宽超过了924、1232，则将原始图片缩放一定比例，使其能够放入到背景上，并略有空余，继续按上面的方式处理，得到两张新图片。
"""

import cv2
import numpy as np
import random
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, NamedTuple
import json


class ResizedImage(NamedTuple):
    x1: int  # 在背景图中的左上角x坐标
    y1: int  # 在背景图中的左上角y坐标
    proc_w: int  # 粘贴在背景图上的图片的宽度
    proc_h: int  # 粘贴在背景图上的图片的高度
    scale: float  # 图片的缩放比例
    whole_image: np.ndarray  # 连同背景在一起的整个图片


class ImageProcessor:
    def __init__(self, bg_width: int = 924, bg_height: int = 1232):
        """
        初始化图片处理器

        Args:
            bg_width: 背景图片宽度
            bg_height: 背景图片高度
        """
        self.bg_width = bg_width
        self.bg_height = bg_height
        self.bg_size = (bg_width, bg_height)

    def _generate_random_background(self) -> np.ndarray:
        """
        生成白色到灰色之间的随机背景

        Returns:
            随机颜色的背景图片
        """
        # 生成200-255之间的随机灰度值（从浅灰到白）
        gray_value = random.randint(200, 255)
        background = np.full(
            (self.bg_height, self.bg_width, 3),
            (gray_value, gray_value, gray_value),
            dtype=np.uint8,
        )
        return background

    def _resize_if_needed(
        self, image: np.ndarray, margin: int = 20
    ) -> Tuple[np.ndarray, float]:
        """
        如果需要，缩放图片使其能放入背景

        Args:
            image: 输入图片
            margin: 边缘留空大小

        Returns:
            (缩放后的图片, 缩放比例)
        """
        h, w = image.shape[:2]
        scale = 1.0

        # 检查是否需要缩放
        if w > self.bg_width - margin or h > self.bg_height - margin:
            # 计算缩放比例，留出margin的边距
            scale_w = (self.bg_width - margin) / w
            scale_h = (self.bg_height - margin) / h
            scale = min(scale_w, scale_h)

            # 计算新尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 缩放图片
            resized_image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            return resized_image, scale

        return image, scale

    def _calculate_max_offset(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        """
        计算最大可偏移量

        Args:
            img_width: 图片宽度
            img_height: 图片高度

        Returns:
            (最大x偏移, 最大y偏移)
        """
        max_x = max(0, self.bg_width - img_width)
        max_y = max(0, self.bg_height - img_height)
        return max_x, max_y

    def process_image(
        self,
        image_path: str,
        random_offset: bool,
        output_dir: str = "processed_images",
    ) -> Dict:
        """
        处理单张图片

        Args:
            image_path: 输入图片路径
            random_offset: 是否使用随机偏移,False表示从左上角（0，0）位置开始
            output_dir: 输出目录

        Returns:
            包含处理信息的字典
        """
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)

        # 读取图片
        image = cv2.imread(image_path)

        # 转换颜色空间（OpenCV默认BGR转RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取原始尺寸
        orig_h, orig_w = image_rgb.shape[:2]

        # 缩放图片（如果需要）
        processed_image, scale = self._resize_if_needed(image_rgb)
        proc_h, proc_w = processed_image.shape[:2]

        # 准备结果信息
        base_name = Path(image_path).stem
        result_info = {
            "original_size": (orig_w, orig_h),
            "processed_size": (proc_w, proc_h),
            "scale_factor": scale,
            "background_size": self.bg_size,
            "images": [],
        }

        # 生成随机背景
        background = self._generate_random_background()

        # 计算偏移量
        if not random_offset:
            # 情况1：左上角开始
            x_offset, y_offset = 0, 0
        else:
            # 情况2：随机偏移（如果有空间）
            max_x, max_y = self._calculate_max_offset(proc_w, proc_h)
            if max_x > 0 and max_y > 0:
                x_offset = random.randint(0, max_x)
                y_offset = random.randint(0, max_y)
            else:
                x_offset, y_offset = 0, 0

        # 如果期望随机偏移，但偏移量为0，则表示失败
        if random_offset and x_offset == 0 and y_offset == 0:
            pass  # 忽略

        # 将图片粘贴到背景上
        result_image = background.copy()
        result_image[
            y_offset : y_offset + proc_h, x_offset : x_offset + proc_w
        ] = processed_image

        # 转换回BGR保存
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        return ResizedImage(
            x1=x_offset,
            y1=y_offset,
            proc_w=proc_w,
            proc_h=proc_h,
            scale=scale,
            whole_image=result_bgr,
        )

    def process_folder(
        self, input_dir: str, output_dir: str = "processed_images"
    ) -> Dict:
        """
        处理整个文件夹中的图片

        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径

        Returns:
            所有图片的处理信息
        """
        input_path = Path(input_dir)
        all_results = {}

        # 支持的图片格式
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        }

        # 遍历所有图片文件
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"处理图片: {image_file.name}")
                try:
                    result = self.process_image(str(image_file), output_dir)
                    all_results[image_file.name] = result
                except Exception as e:
                    print(f"处理图片 {image_file.name} 时出错: {str(e)}")

        # 保存处理信息到JSON文件
        info_path = os.path.join(output_dir, "processing_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n所有图片处理完成！处理信息已保存到: {info_path}")
        return all_results


def main():
    # 创建处理器实例
    processor = ImageProcessor(bg_width=924, bg_height=1232)

    # 使用示例1：处理单张图片
    # result = processor.process_image("input.jpg", "processed_images")

    # 使用示例2：处理整个文件夹
    processor.process_folder("input_images", "processed_images")


if __name__ == "__main__":
    main()
