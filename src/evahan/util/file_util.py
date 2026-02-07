from pathlib import Path


def list_image_files(folder: Path, recursive: bool = False) -> list[Path]:
    """列出文件夹下所有图片文件

    Args:
        folder (Path): 文件夹路径

    Returns:
        list[Path]: 图片文件列表
    """
    generator = folder.rglob("*") if recursive else folder.glob("*")
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]

    files = sorted(
        [f for f in generator if f.suffix.lower() in image_extensions]
    )
    return [f for f in files]
