from typing import Tuple, List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_faiss_index(texts: List[str]) -> Tuple[SentenceTransformer, faiss.IndexFlatIP]:

    model = SentenceTransformer(MODEL_NAME)

    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    return model, index
