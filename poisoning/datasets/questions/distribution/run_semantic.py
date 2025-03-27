"""Get the number of high cosine similarity candidate for each question"""
from tqdm import tqdm
import numpy as np
import torch
import json


THRESHOLD = 0.8
datasets = [
    'microsoft/ms_marco',
    'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]
