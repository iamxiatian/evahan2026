"""
采用OpenAI的方式，调用本地部署的模型服务进行图文问答。

需要提前运行：./scripts/deploy.sh

"""

from openai import OpenAI
from evahan import config

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8002/v1",
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

def query(image_url: str, query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query},
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model_type, messages=messages, seed=42
    )
    response = resp.choices[0].message.content
    return response

if __name__ == "__main__":
    image_url = test_image = "/data/app/workspace/evahan2026/dataset/train_data/Dataset_A/a_0001.jpg"
    response = query(image_url, config.OCR_QUERY)
    print(response)
