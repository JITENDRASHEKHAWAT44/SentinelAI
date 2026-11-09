from backend.database.vector_store import retrieve_documents, retrieve_documents_with_scores


def _relevance_from_distance(distance: float) -> float:
    """Convert vector distance into a 0-1 relevance score for the UI."""
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, 1.0 / (1.0 + value))), 4)


def retrieve_context(query: str, k: int = 5):

    try:
        scored = retrieve_documents_with_scores(query=query, k=k)
        retrieved_chunks = []
        for doc, distance in scored:
            retrieved_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "score": float(distance) if distance is not None else None,
                "relevance": _relevance_from_distance(distance),
            })
        return retrieved_chunks
    except Exception:
        documents = retrieve_documents(query=query, k=k)
        if not documents:
            return []
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "score": None,
                "relevance": None,
            }
            for doc in documents
        ]