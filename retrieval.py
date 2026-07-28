from sentence_transformers import CrossEncoder
from config import RERANK_MODEL, RERANK_CANDIDATES, RERANK_TOP_K

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder

def retrieve_chunks(query, model, collection, source_filter=None):
    query_embedding = model.encode([query])[0].tolist()

    where_filter = {}
    if source_filter:
        where_filter["source"] = source_filter

    # stage 1 — bi-encoder: get 50 candidates from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RERANK_CANDIDATES,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )

    candidates = []
    for i in range(len(results["documents"][0])):
        candidates.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    if not candidates:
        return []

    # stage 2 — cross-encoder: score each [query, chunk] pair together
    cross_encoder = get_cross_encoder()
    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = cross_encoder.predict(pairs)

    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = float(scores[i])

    # sort by reranker score descending, keep top 5
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:RERANK_TOP_K]