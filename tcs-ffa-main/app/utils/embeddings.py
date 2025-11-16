
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model

def embed_texts(texts):
    model = load_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)
