import time
import sys
gptcache_path = '/home/taojie_wang/GPTCache'
sys.path.insert(0, gptcache_path)

import gptcache
print(gptcache.__file__)


data_dir = '/home/taojie_wang/GPTCache/multimodal'
import shutil
import os
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)


import torch
from PIL import ImageChops
from diffusers import DPMSolverMultistepScheduler
from gptcache.adapter.diffusers import StableDiffusionPipeline
from gptcache.core import Cache, Config
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import (
    Huggingface,
)
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# onnx = Onnx()
# cache_base = CacheBase('sqlite')
# vector_base = VectorBase('faiss', dimension=onnx.dimension, index_path='./local/faiss20.index'),
# print(f"type of vector base: {type(vector_base)}")
# vector_base = vector_base[0]
# object_base = ObjectBase('local', path='./images')
# data_manager = get_data_manager(cache_base, vector_base, object_base)

# # this is the gloval cache variable
# cache.init(
#     pre_embedding_func=get_prompt,
#     embedding_func=onnx.to_embeddings,
#     data_manager=data_manager,
#     similarity_evaluation=SearchDistanceEvaluation(),
#     )


the_cache = Cache()
embedding=Huggingface()
data_manager = manager_factory(
            "sqlite,faiss,local",
            data_dir=data_dir,
            vector_params={"dimension": embedding.dimension, "top_k": 5},
        )
init_similar_cache(
    data_dir=data_dir,
    cache_obj=the_cache,
    embedding=embedding,
    data_manager=data_manager,
    evaluation=SearchDistanceEvaluation(),
    config=Config(similarity_threshold=0.6),
)



model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# prompt = "a photo of a cat. DO NOT GENERATE THIS: a photo of an astronaut riding a horse on mars"
prompt = "a photo of a cat.  a photo of an astronaut riding a horse on mars"
start = time.time()
image1 = pipe(prompt=prompt, cache_obj=the_cache).images[0]
print("First time generation:", time.time() - start)

start = time.time()
image2 = pipe(prompt=prompt, cache_obj=the_cache).images[0]
print("Second time generation:", time.time() - start)

# Compare generated images
diff = ImageChops.difference(image1, image2)
assert not diff.getbbox(), "Got different images."

# # save images
# image1.save("/home/taojie_wang/GPTCache/multimodal/images/astronaut_horse_mars_1.png")
# image2.save("/home/taojie_wang/GPTCache/multimodal/images/astronaut_horse_mars_2.png")

print("we are done")