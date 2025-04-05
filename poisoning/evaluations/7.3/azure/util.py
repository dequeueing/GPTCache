import redis
from langchain_openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache


AZURE_ENDPOINT_TURBO="https://32908-m7u0bg59-swedencentral.cognitiveservices.azure.com/"
AZURE_ENDPOINT_EMBED="https://32908-m7slni1z-eastus2.cognitiveservices.azure.com/"
API_KEY_EMBED="9ZJ7Ejcd2S7CHgmnWNYxPB4BUX5AYrA3ciNDPM5hEtnLLaQ0D0kWJQQJ99BCACHYHv6XJ3w3AAAAACOGtL8h"
API_KEY_TURBO="GA6q4dISaWMZfyFW3GfzLe7fZfgVctzQQL1cYMkfGHFp0A4tIo9jJQQJ99BCACfhMk5XJ3w3AAAAACOG6Acp"

API_VERSION="2023-05-15"

LLM_DEPLOYMENT_NAME="gpt-35-turbo-instruct"
LLM_MODEL_NAME="gpt-35-turbo-instruct"

EMBEDDINGS_DEPLOYMENT_NAME="text-embedding-ada-002"
EMBEDDINGS_MODEL_NAME="text-embedding-ada-002"

# old redis, filled with 800 noise questions
# REDIS_ENDPOINT = "my-managed-redis-semantic-new.eastus.redis.azure.net:10000"
# REDIS_PASSWORD = "LeFV2ZOgLCGNBCbRJGCLJehEv4Vd2NXN9AzCaMSRd5s="

# new redis cache, empty
REDIS_ENDPOINT = "empty-semantic-cache.eastus.redis.azure.net:10000"
REDIS_PASSWORD = "W4QUlWGJp4wCNU9agawnq69K94AyWtnyuAzCaLc41F0="


def delete_cache_entry_by_prompt(prompt, redis_client):
    cursor = '0'
    found = False
    while True:  # Keep iterating until SCAN returns cursor 0 (end of scan)
        cursor, keys = redis_client.scan(cursor=cursor, match="doc:cache:*", count=100)
        for key in keys:
            stored_prompt = redis_client.hget(key, "prompt")
            if stored_prompt == prompt:
                redis_client.delete(key)
                print(f"Deleted cache entry for prompt: {prompt} (key: {key})")
                found = True
        if cursor == 0:  # Exit only when SCAN has fully traversed the keyspace
            break
    if not found:
        print(f"No cache entry found for prompt: {prompt}")


# Step 3: List all keys before deletion
def scan_cache(redis_client):
  cursor = '0'
  while cursor != 0:
      cursor, keys = redis_client.scan(cursor=cursor, match="doc:cache:*")
      for key in keys:
          prompt = redis_client.hget(key, "prompt")
          print(f"Key: {key}, Prompt: {prompt}")


# delete_cache_entry_by_prompt(attacker_prompt, redis_client)

# make sure you have an LLM deployed in your Azure Open AI account. In this example, I used the GPT 3.5 turbo instruct model. My deployment was named "gpt35instruct".
llm = AzureOpenAI(
    deployment_name=LLM_DEPLOYMENT_NAME,
    model_name="gpt-35-turbo-instruct",
    openai_api_key=API_KEY_TURBO,
    azure_endpoint=AZURE_ENDPOINT_TURBO,
    openai_api_version=API_VERSION,
    max_tokens=10,
)
# make sure you have an embeddings model deployed in your Azure Open AI account. In this example, I used the text embedding ada 002 model. My deployment was named "textembedding".
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    model="text-embedding-ada-002",
    openai_api_key=API_KEY_EMBED,
    azure_endpoint=AZURE_ENDPOINT_EMBED,
    openai_api_version=API_VERSION,
    chunk_size=2048
)

# create a connection string for the Redis Vector Store. Uses Redis-py format: https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url
# This example assumes TLS is enabled. If not, use "redis://" instead of "rediss://
redis_url = "rediss://:" + REDIS_PASSWORD + "@"+ REDIS_ENDPOINT
set_llm_cache(RedisSemanticCache(redis_url = redis_url, embedding=embeddings, score_threshold=0.2))
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
