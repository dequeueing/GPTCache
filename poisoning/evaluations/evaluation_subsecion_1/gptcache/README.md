# Setup

## Datasets

```python
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
```

Randomly choose 10 questions each round, and repeat ten rounds. So 100 questions for each dataset.

For each question, generate a wrong answer using LLM (Llama3 instruct). Then craft white and black box prompts.

## Hyperparameter
1. embedding model: distilbert-base-uncased
2. semantic evaluation: cross-encoder/quora-distilrobert
3. threshold: 0.8
4. noise: none
5. top_k to retrieve from database: 5
6. vector database: sqlite,faiss
