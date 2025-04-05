import openai
import os
import langchain
import time
from util import redis_client
from util import llm
from util import scan_cache


def generate(prompt):
  response = llm(prompt)
  return response.strip()


if __name__ == '__main__':
    scan_cache(redis_client)
    completion = generate("what is your name?")