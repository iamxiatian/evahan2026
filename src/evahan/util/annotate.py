from pathlib import Path

from PIL import Image, ImageDraw

from evahan.core import EvahanRegion


color_map: dict[str, tuple[int, int, int]] = {
    "book_edge": (255, 0, 0),  # 红色
    "image": (0, 255, 0),  # 绿色
    "seal": (0, 0, 255),  # 蓝色
    "text": (255, 165, 0),  # 橙色
}


def contrast_color(
    rgb: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    考虑多种对比度因素的改进算法, rgb的颜色值是0-1之间的浮点数, 返回值也是0-1之间的浮点数
    """
    # 将0-1范围的rgb转回0-255
    r, g, b = (int(c * 255) for c in rgb)
    # 基于0-255计算亮度(更符合人眼感知)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # 亮度归一化到0-1
    luminance /= 255.0
    contrast_with_black = luminance
    contrast_with_white = 1 - luminance

    text_color: tuple[float, float, float] = (
        (0, 0, 0) if contrast_with_black > contrast_with_white else (1, 1, 1)
    )
    # 在调用contrast_color前添加
    # print(f"背景色(RGB 0-1): {rgb} → 转换为0-255: {[c*255 for c in rgb]}")
    # print(f"计算出的文字色: {text_color}")

    return text_color


def draw_element(
    image: Image.Image,
    label: str,
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    p4: tuple[int, int],
) -> Image.Image:
    """
    在图像上标注元素
    Args:
        image: PIL图像对象
        label: 元素标签
        points: 元素边界框坐标列表, 顺序为左上、右上、右下、左下
    """
    # 创建可绘制对象
    draw = ImageDraw.Draw(image, mode="RGBA")  # 启用alpha通道支持透明度
    # 1. 绘制边框(与PDF的rectangle保持同色同宽)
    color: tuple[int, int, int] = color_map[label]
    draw.polygon(
        [p1, p2, p3, p4],
        outline=color,
        width=2,
    )

    # 2. 绘制文字背景
    # 文字框左上角与大框左上角对齐(核心:x=x1, y=y1)
    text_x, text_y = p1

    # 1. 计算文字的实际边界框(基于当前字体和位置)
    text_bbox = draw.textbbox((text_x, text_y), label)
    # text_bbox格式:(left, top, right, bottom)

    # 2. 扩展文字背景框(保留边距, 让背景比文字稍大)
    # 上下左右各加5像素边距(可根据需要调整)
    bg_padding = 5
    bg_rect = (
        p1[0],  # 左边界
        p1[1],  # 上边界(向上扩展)
        text_bbox[2] + bg_padding * 2,  # 右边界(向右扩展)
        text_bbox[3] + bg_padding,  # 下边界(向下扩展)
    )

    # 3. 绘制背景框(从左上角开始, 与大框紧密贴合)
    draw.rectangle(
        bg_rect,
        fill=(*color, 175),  # 带透明度的背景色
        outline=None,
    )

    # 4. 绘制文字(在背景框内居中微调, 避免贴边)
    r, g, b = color
    text_color = (r / 255.0, g / 255.0, b / 255.0)
    text_color = contrast_color(text_color)
    #  转换为0-255整数供ImageDraw使用
    text_color = tuple(int(c * 255) for c in text_color)

    draw.text(
        (text_x + bg_padding, text_y),  # 文字左上角与大框左上角对齐
        label,
        fill=text_color,
    )

    return image


def visualize_layout(
    image_path: Path, regions: list[EvahanRegion], save_path: Path
) -> None:
    """在图片上绘制版面元素区域，用于EvaHan2026版面元素的可视化验证。
    Args:
        image_path (Path): 图片路径
        regions (list[EvahanRegion]): 版面元素区域列表
        save_path (str): 输出图片路径
    """
    image: Image.Image = Image.open(image_path).convert("RGB")
    for region in regions:
        image = draw_element(
            image=image,
            label=region.label,
            p1=region.points[0],
            p2=region.points[1],
            p3=region.points[2],
            p4=region.points[3],
        )

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(save_path)
