"""
可视化元素布局，在图片上绘制识别的布局元素。

Example:
    ```shell
    uv run -m evahan.viz_layout lora_testset
    ```
"""

import json
from pathlib import Path

from rich import print
from rich.progress import track

from evahan import config
from evahan.core import EvahanRegion
from evahan.extract import extract_layout_regions
from evahan.util import file_util
from evahan.util.annotate import visualize_layout


def draw_layout(
    image_folder: Path,
    save_folder: Path,
    layout_dict: dict[str, list[EvahanRegion]],
) -> None:
    """处理数据集中的所有图像，并将可视化结果保存到指定目录。
    Args:
        image_folder (Path): 包含图像文件的文件夹路径。
        save_folder (Path): 可视化结果保存的文件夹路径。
        layout_dict (dict[str, list[EvahanRegion]]): 图像路径到布局区域列表的映射，其中可以的形式为"Dataset_A/a_0001.jpg"。
    """
    if not image_folder.exists():
        print(f"[red]待处理的图片文件夹不存在：{image_folder}[/red]")
        return

    image_files = file_util.list_image_files(image_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for image_file in track(image_files, description="draw layout..."):
        img_path: str = f"{image_folder.name}/{image_file.name}"
        save_path: Path = save_folder / image_file.name

        if img_path not in layout_dict:
            print(f"[red]Warning: {img_path} not in layout_dict[/red]")
            continue
        regions: list[EvahanRegion] = layout_dict[img_path]

        visualize_layout(
            image_file,
            regions,
            save_path=save_path,
        )


def draw_testset_results(
    run_name: str,
) -> None:
    """把某一次测试集运行结果的版面布局可视化出来。
    Args:
        run_name (str): 对测试集的一次运行结果的名称。
    """
    base_folder = config.EVAHAN_DATA_PATH / "run_results" / run_name
    json_file = base_folder / "Task_B.json"
    regions_dict: dict[str, list[EvahanRegion]] = {}
    with json_file.open("r", encoding="utf-8") as f:
        items = json.load(f)
        for item in items:
            image_path = item["image_path"]
            llm_response = item["llm_response"]
            regions = extract_layout_regions(llm_response)
            regions_dict[image_path] = regions

    image_folder = config.EVAHAN_TESTSET_PATH / "Task_B"
    save_folder = base_folder / "Task_B_annotated"
    draw_layout(
        image_folder=image_folder,
        save_folder=save_folder,
        layout_dict=regions_dict,
    )


if __name__ == "__main__":
    import typer

    typer.run(draw_testset_results)
