from typing import Union, Dict, Tuple, List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
import json

class SemanticEvaluator:
    def __init__(self, batch_size: int = 512, device: str = None):
        """
        Initialize the SemanticEvaluator with optional device specification.
        
        Args:
            batch_size: Number of pairs to process in one batch
            device: Specific device to use ('cuda', 'cpu', etc.), defaults to auto-detection
        """
        model_id = "cross-encoder/quora-distilroberta-base"
        self.model_id = model_id
        
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize CrossEncoder with specified device
        self.encoder = CrossEncoder(model_id, device=self.device)
        self.batch_size = batch_size
        
    def predict(self, q1: Union[str, List[str]], q2: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """Predict semantic similarity scores for pairs of questions.
        
        Args:
            q1: Single question or list of questions
            q2: Single question or list of corresponding questions
        
        Returns:
            Single score (float) or array of scores (np.ndarray) depending on input
        """
        # Handle single pair case
        if isinstance(q1, str) and isinstance(q2, str):
            score = self.encoder.predict([(q1, q2)])[0]
            return float(score)
        
        # Handle batch case
        elif isinstance(q1, list) and isinstance(q2, list):
            if len(q1) != len(q2):
                raise ValueError("q1 and q2 lists must have the same length")
            
            # Create list of tuples for batch processing
            sentence_pairs = list(zip(q1, q2))
            
            # Process in batches
            scores = []
            for i in tqdm(range(0, len(sentence_pairs), self.batch_size), desc="Predicting scores"):
                batch = sentence_pairs[i:i + self.batch_size]
                batch_scores = self.encoder.predict(batch, convert_to_numpy=True)
                scores.extend(batch_scores)
            
            return np.array(scores)
        
        else:
            raise TypeError("q1 and q2 must both be strings or both be lists")

# Example usage
if __name__ == "__main__":
    # Initialize with automatic device detection
    evaluator = SemanticEvaluator(batch_size=32)
    
    # Single prediction
    score = evaluator.predict("How are you?", "Are you okay?")
    print(f"Single score: {score}")
    
    # Batch prediction
    q1_list = ["How are you?", "What's the weather like?", "Where are you going?"]
    q2_list = ["Are you okay?", "Is it sunny today?", "What’s your destination?"]
    scores = evaluator.predict(q1_list, q2_list)
    print(f"Batch scores: {scores}")