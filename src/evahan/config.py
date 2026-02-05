import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from evahan import prompt


logger = structlog.get_logger()

load_dotenv()

# Evahan数据集的存放路径
EVAHAN_DATA_PATH: Path = Path(
    os.getenv("evahan_dataset_parent_path", default="./dataset")
)

# 训练集的父路径
EVAHAN_TRAINSET_PATH: Path = EVAHAN_DATA_PATH / "train_data"

# 三个数据集的路径
EVAHAN_TRAINSET_A: Path = EVAHAN_TRAINSET_PATH / "Dataset_A"
EVAHAN_TRAINSET_B: Path = EVAHAN_TRAINSET_PATH / "Dataset_B"
EVAHAN_TRAINSET_C: Path = EVAHAN_TRAINSET_PATH / "Dataset_C"

# 测试数据集的目录
EVAHAN_TESTSET_PATH: Path = EVAHAN_DATA_PATH / "test_data"


if EVAHAN_DATA_PATH.exists():
    logger.info(f"EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH}")
else:
    logger.error(f"EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH} not exists!")

# OCR的默认提取提示语
OCR_USER_QUERY = prompt.OCR_USER_QUERY_V1

# 版面默认的提示
LAYOUT_USER_QUERY = prompt.LAYOUT_USER_QUERY_V1
