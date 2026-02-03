"""可视化版面数据集"""

from pathlib import Path

from evahan.dataset import load_evahan_layout_dataset
from evahan.util.annotate import visualize_layout


def visualize_layout_dataset(
    json_file: Path, dataset_path: Path, out_path: Path
):
    """可视化版面数据集，保存到指定路径
    Args:
        json_file (Path): Evahan版面数据集的json文件路径
        dataset_path (Path): Evahan版面数据集图片所在父目录，如`train_data`目录，json_file中的image_path是相对于该目录的相对路径.
        out_path (Path): 可视化图片保存路径
    """

    # 注意，因为json_file不一定存放在train_data目录下，因此返回对象中的iamge_path不能直接使用，而是需要利用其相对路径，拼接`dataset_path`获取真正存在的绝对路径
    layout_items = load_evahan_layout_dataset(json_file)
    for item in layout_items:
        img_path = dataset_path / item.relative_image_path
        visualize_layout(
            img_path,
            item.regions,
            save_path=out_path / f"{img_path.stem}_layout_viz{img_path.suffix}",
        )
