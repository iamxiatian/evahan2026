"""
采用OpenAI的方式，调用本地部署的模型服务进行图文问答。

需要提前运行：./scripts/deploy.sh

"""

from pathlib import Path

from openai import OpenAI

from evahan import config


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
        model=model_type, messages=messages, seed=42
    )
    response = resp.choices[0].message.content
    return response


def test_ocr():
    image_path = config.EVAHAN_TRAIN_PATH_A / "a_0001.jpg"
    response = query(image_path.as_posix(), config.OCR_USER_QUERY)
    print(response)


def test_layout():
    image_path: Path = config.EVAHAN_TRAIN_PATH_B / "b_0001.jpg"

    response = query(image_path.as_posix(), config.LAYOUT_USER_QUERY)
    print(response)


if __name__ == "__main__":
    test_layout()
    test_ocr()
