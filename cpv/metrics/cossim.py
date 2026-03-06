import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_score(emb1, emb2):
    # Convert tensors to numpy arrays if necessary
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.detach().cpu().numpy()
    
    # Ensure both embeddings are 2D
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    
    return cosine_similarity(emb1, emb2)[0][0]