import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def convert_bw_or_gray(
    input_path: Path,
) -> tuple[MatLike, Literal["bw"] | Literal["gray"]]:
    """
    将图片尝试转换为白底黑字，如果无法分离，则转换为灰度图,并返回处理后的图片内容

    Args:
        input_path: 输入图片路径
    Returns:
        MatLike: 处理后的图片内容
        Literal["bw"] | Literal["gray"]: 图片是否转换为白底黑字或灰度图
    """
    # 1. 读取图片并转灰度图
    img = cv2.imread(input_path.as_posix())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 预处理：分析图片灰度分布，动态调整阈值（核心优化）
    # 计算灰度图的关键统计值
    gray_min = np.min(gray)  # 最小灰度值（纯黑=0）
    gray_mean = np.mean(gray)  # 平均灰度值
    black_pixel_ratio = np.sum(gray < 50) / gray.size  # 纯黑像素占比（<50为黑）

    # 判定是否为“有效图片”（有黑色内容）
    is_valid = (gray_min < 50) and (black_pixel_ratio > 0.01)  # 纯黑像素占比>1%
    if not is_valid:
        # 无有效黑色内容，直接保存原图（避免转成全白）
        logger.debug(f"无有效黑色内容，直接使用灰度图:{input_path}")
        return gray, "gray"

    # 3. 动态计算阈值（替代固定22%）
    # 规则：根据平均灰度调整阈值，偏暗图片降低阈值，偏亮图片用原阈值
    if gray_mean < 150:  # 偏暗图片（如深色背景+浅色字/暗书法图）
        threshold_ratio = 0.15  # 阈值调低到15%（≈38），避免字被洗白
    elif gray_mean > 200:  # 偏亮图片（如米黄背景+黑字）
        threshold_ratio = 0.22  # 用原22%阈值（≈56）
    else:  # 中等亮度
        threshold_ratio = 0.18  # 折中阈值18%（≈46）
    threshold_value = 255 * threshold_ratio

    # 4. 二值化（保留黑色内容）
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 5. 校验：转换后是否全白，若是则降低阈值重试
    white_pixel_ratio = np.sum(binary == 255) / binary.size
    if white_pixel_ratio > 0.90:  # 99%以上都是白色，判定为“全白风险”
        logger.debug(f"图片 {input_path} 转换后接近全白，降低阈值重试...")
        # 阈值再降低50%
        threshold_value = threshold_value * 0.5
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        # 若重试后仍全白，直接保存灰度图
        if np.sum(binary == 255) / binary.size > 0.90:
            logger.debug(f"重试后仍全白，使用灰度图：{input_path}")
            return gray, "gray"

    # 6. 温和的white-threshold（避免过度强制纯白）
    white_threshold = 255 * 0.98  # 从95%调高到98%，减少强制纯白的范围
    binary[binary >= white_threshold] = 255
    return binary, "bw"


def convert_with_save(input_path: Path, output_path: Path) -> None:
    """
    将输入图片转为白色或灰色图，并将处理后的图片内容保存到指定路径
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
    """
    # 1. 转换图片为白底黑字或灰度图
    img, _ = convert_bw_or_gray(input_path)
    cv2.imwrite(output_path.as_posix(), img)
    logger.info(f"处理完成，保存至：{output_path}")


def merge_image_vertically(img1: MatLike, img2: MatLike) -> MatLike:
    """
    OpenCV实现两张图片上下合并（自动对齐宽度）
    Args:
        img1: 上方图片数组
        img2: 下方图片数组
    Returns:
        MatLike: 合并后的图片数组
    """
    # 1. 统一宽度（以较宽的图片为基准）
    max_width = max(img1.shape[1], img2.shape[1])
    # 缩放img1
    img1_resized = cv2.resize(
        img1,
        (max_width, int(img1.shape[0] * max_width / img1.shape[1])),
        interpolation=cv2.INTER_LANCZOS4,
    )
    # 缩放img2
    img2_resized = cv2.resize(
        img2,
        (max_width, int(img2.shape[0] * max_width / img2.shape[1])),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # 2. 上下拼接（vstack=vertical stack，垂直堆叠）
    merged_img = np.vstack((img1_resized, img2_resized))

    return merged_img


def merge_file_vertically(
    img1_path: Path,
    img2_path: Path,
    output_path: Path,
    convert_img_bw: bool = True,
) -> None:
    """
    OpenCV实现两张图片上下合并（自动对齐宽度）
    Args:
        img1_path: 上方图片路径
        img2_path: 下方图片路径
        output_path: 合并后图片保存路径
        convert_img_bw: bool = True, 是否将img1和img2转换为白底黑字或灰度图
    """
    # 1. 读取两张图片，根据参数判断是否转换为白底黑字或灰度图

    img1 = (
        convert_bw_or_gray(img1_path)[0]
        if convert_img_bw
        else cv2.imread(img1_path.as_posix())
    )

    img2 = (
        convert_bw_or_gray(img2_path)[0]
        if convert_img_bw
        else cv2.imread(img2_path.as_posix())
    )

    # 2. 上下拼接（vstack=vertical stack，垂直堆叠）
    merged_img = merge_image_vertically(img1, img2)

    # 3. 保存结果
    cv2.imwrite(output_path.as_posix(), merged_img)
    logger.debug(f"图片合并完成，保存至：{output_path}")


def main():
    """测试"""
    from evahan import config

    merge_file_vertically(
        config.EVAHAN_TRAINSET_A / "a_0001.jpg",
        config.EVAHAN_TRAINSET_C / "c_0001.jpg",
        Path("merged.jpg"),
        True,
    )


if __name__ == "__main__":
    main()
