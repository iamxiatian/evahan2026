"""准备数据集，方便从头开始处理"""

import os
import shutil

import typer
from rich import print

from evahan.util.image_rotate import rotate_folder
from evahan.convert import convert_to_swift

def prepare_dataset(evahan_zip_path: str):
    """解压并准备Evahan数据集"""
    extract_to: str = "./dataset"
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"正在解压数据集 {evahan_zip_path} ...")
    shutil.unpack_archive(evahan_zip_path, extract_to)
    print(f"数据集已解压到 {extract_to}")

    # 数据集A和数据集C需要执行旋转处理
    dataset_A = os.path.join(extract_to, "train_data", "Dataset_A")
    dataset_C = os.path.join(extract_to, "train_data", "Dataset_C")
    rotate_folder(dataset_A, dataset_A)  # 原地旋转
    print("Dataset_A 旋转完成")
    rotate_folder(dataset_C, dataset_C)  # 原地旋转
    print("Dataset_C 旋转完成")
    
    # 生成swift格式的训练数据
    print("正在生成swift格式的数据...")
    convert_to_swift(format="jsonl", use_abs_img_path=True)
    convert_to_swift(format="json", use_abs_img_path=True)
    
    print("数据集准备完成！")


if __name__ == "__main__":
    print("Usage: uv run -m evahan.prepare <path_to_evahan_zip>")
    typer.run(prepare_dataset)
    