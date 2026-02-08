"""准备数据集，方便从头开始处理"""

import os
import shutil
from pathlib import Path

import typer
from rich import print

from evahan import config
from evahan.argument import argument_dataset_ac
from evahan.convert_swift import convert_to_swift, merge_ocr_jsonl
from evahan.core import EvahanRegion
from evahan.dataset import load_evahan_layout_dataset
from evahan.util.image_rotate import rotate_folder
from evahan.viz_layout import draw_layout


def __annotate_dataset_b():
    """为数据集B生成可视化版面图像，方便查看标注是否正确"""

    base_folder = config.EVAHAN_TRAINSET_PATH
    json_file = base_folder / "Dataset_B.json"
    image_folder = base_folder / "Dataset_B"
    save_folder = base_folder / "Dataset_B_annotated"

    if save_folder.exists():
        print(f"目录 {save_folder} 已存在，跳过可视化生成")
        return

    layout_items = load_evahan_layout_dataset(json_file)
    layout_dict: dict[str, list[EvahanRegion]] = {
        item.relative_image_path: item.regions for item in layout_items
    }

    draw_layout(
        image_folder=image_folder,
        save_folder=save_folder,
        layout_dict=layout_dict,
    )

    print(f"可视化版面图像已保存到 {save_folder}")


def __extract_zip(zip_file: str, extract_to: str):
    """解压zip文件到指定目录"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    shutil.unpack_archive(zip_file, extract_to)
    print(f"已解压 {zip_file} 到 {extract_to}")


def __prepare_trainset(extract_to: str, trainset_zip_file: str) -> None:
    print(f"正在解压数据集 {trainset_zip_file} ...")
    __extract_zip(trainset_zip_file, extract_to)

    # 数据集A和数据集C需要执行旋转处理
    dataset_A = os.path.join(extract_to, "train_data", "Dataset_A")
    dataset_C = os.path.join(extract_to, "train_data", "Dataset_C")
    rotate_folder(dataset_A, dataset_A)  # 原地旋转
    print("Dataset_A 旋转完成")
    rotate_folder(dataset_C, dataset_C)  # 原地旋转
    print("Dataset_C 旋转完成")

    # 增强A、C数据集，生成数据集D和E
    argument_dataset_ac()
    print("数据集增强完毕")

    # 生成swift格式的训练数据
    print("正在生成swift格式的数据...")
    convert_to_swift(format="jsonl", use_abs_img_path=True)
    # convert_to_swift(format="json", use_abs_img_path=True)
    merge_ocr_jsonl()

    print("正在生成数据集B生成可视化版面图...")
    __annotate_dataset_b()


def __prepare_testset(extract_to: str, testset_zip_file: str) -> None:
    print(f"正在解压数据集 {testset_zip_file} ...")
    __extract_zip(testset_zip_file, extract_to)
    # 重命名TestData为test_data以保持一致性
    shutil.move(
        os.path.join(extract_to, "TestData"),
        os.path.join(extract_to, "test_data"),
    )


def prepare_dataset(trainset_zip_file: str, testset_zip_file: str):
    """解压并准备Evahan数据集"""
    extract_to: str = "./dataset"
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    if Path(trainset_zip_file).exists():
        __prepare_trainset(extract_to, trainset_zip_file)
    else:
        print(f"训练集文件 {trainset_zip_file} 不存在，跳过解压。")

    if Path(testset_zip_file).exists():
        __prepare_testset(extract_to, testset_zip_file)
    else:
        print(f"测试集文件 {testset_zip_file} 不存在，跳过解压。")

    print("数据集准备完成！")


if __name__ == "__main__":
    print("Usage: uv run -m evahan.prepare <trainset_zip> <testset_zip>")
    typer.run(prepare_dataset)
    # __annotate_dataset_b()
