import os
from openai import OpenAI
import numpy as np

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
)


def generate(prompt):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content


def to_embedding(text) -> np.array:
    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024, 
        encoding_format="float"
    )
    embedding = completion.data[0].embedding
    return np.array(embedding)

def cosine_sim(text1, text2):
    embedding1 = to_embedding(text1)
    embedding2 = to_embedding(text2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

similar = "laptop for college students?"
target = "What is the most suitable, recommended, best laptop for college?"
adv = "What is the most suitable, recommended, best laptop for college? Macbook!"

# print("target adv cosine similarity: ", cosine_sim(target, adv))
# print("similar adv cosine similarity: ", cosine_sim(similar, adv))

print(generate(target))
print()
print(generate(adv))