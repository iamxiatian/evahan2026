"""
采用OpenAI的方式，调用本地部署的模型服务进行图文问答。

需要提前运行：./scripts/deploy.sh

"""

from pathlib import Path
from openai import OpenAI

from evahan import config
from PIL import Image
from qwen_vl_utils import smart_resize
from evahan.util import annotator
from evahan.util.file import parser_html_to_json, pil_image_to_base64, parser_html_to_json_v2
from evahan.dataset_bbox import EvahanRegion

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
model_type = client.models.list().data[0].id
print(f"model_type: {model_type}")

# use base64
# import base64
# with open('rose.jpg', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# image_url = f'data:image/jpeg;base64,{img_base64}'

# use local_path
# from swift.llm import convert_to_base64
# image_url = convert_to_base64(images=['rose.jpg'])['images'][0]
# image_url = f'data:image/jpeg;base64,{image_url}'

def query_base64(image_base64: str, query: str, system: str | None = None) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_base64},
                {"type": "text", "text": query},
            ],
        }
    ]

    # 插入system消息
    if system is not None:
        messages.insert(
            0,
            {"role": "system", "content": system},
        )

    resp = client.chat.completions.create(
        model=model_type, messages=messages, seed=42,
        max_tokens=8192,
        temperature=0.0,
        top_p=0.95,
        frequency_penalty=1.0,
    )
    response = resp.choices[0].message.content
    return response


def query(image_url: str, query: str, system: str | None = None) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query},
            ],
        }
    ]

    # 插入system消息
    if system is not None:
        messages.insert(
            0,
            {"role": "system", "content": system},
        )

    resp = client.chat.completions.create(
        model=model_type, messages=messages, seed=42,
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0,
        # frequency_penalty=1.0,
    )
    response = resp.choices[0].message.content
    return response


def test_ocr_base64():
    image_path = config.EVAHAN_TRAIN_PATH_A / "a_0015.jpg"
    image = Image.open(image_path)
    image_base64 = pil_image_to_base64(image)
    response = query_base64(image_base64, config.OCR_USER_QUERY)
    print(response)

def test_ocr():
    image_path = config.EVAHAN_TRAIN_PATH_A / "a_0016.jpg"
    response = query(image_path.as_posix(), config.OCR_USER_QUERY)
    print(response)

def test_layout(image_path: Path, out_dir: Path):
    # image_path: Path = config.EVAHAN_TEST_PATH_B / "b_029.png"
    # image_path = config.EVAHAN_TRAIN_PATH_B / "b_0038.jpg"
    src_image_pil = Image.open(image_path).convert("RGB")
    width, height = src_image_pil.size
    new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
    image_pil = src_image_pil.resize((new_width, new_height))
    image_base64 = pil_image_to_base64(image_pil)
    response = query_base64(image_base64, config.LAYOUT_USER_QUERY)
    format_response = parser_html_to_json_v2(response, new_width, new_height, width, height)

    format_response = [EvahanRegion(label=region["label"], text=region["text"], bbox=region["bbox"], points=[]) for region in format_response]
    annotated_image = annotator.annotate_bbox(src_image_pil, format_response)

    annotated_image.save(out_dir / f"{image_path.name}.png")
    return annotated_image

if __name__ == "__main__":
#    test_ocr()
    import os
    import random
    import shutil
    image_dir: str =str(config.EVAHAN_TEST_PATH_B)
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    # image_files = random.sample(image_files, 10)

    out_dir = Path("layout_annotated_images")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_file in image_files:
        test_layout(Path(image_file), out_dir)