import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    if len(docs) == 0:
        return np.array([], dtype=float)

    N = len(docs)

    # Document lengths
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths)

    # Document frequency (df)
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = np.zeros(N, dtype=float)

    for i, doc in enumerate(docs):
        tf = Counter(doc)
        dl = doc_lengths[i]

        score = 0.0

        for term in query_tokens:
            if df.get(term, 0) == 0:
                continue

            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            freq = tf.get(term, 0)
            if freq == 0:
                continue

            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf * (freq * (k1 + 1)) / denom

        scores[i] = score

    return scores