from typing import Optional
import numpy as np
import pandas as pd

def retrieve_top_k(
    df: pd.DataFrame,
    query: str,
    model,
    index,
    k: int = 5
) -> pd.DataFrame:
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, idxs = index.search(q, k)

    hits = df.iloc[idxs[0]].copy()
    hits["similarity"] = scores[0]
    return hits
