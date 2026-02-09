import os
from typing import List, Tuple
from pinecone import Pinecone

BOOK_INDEX_NAME = "book-metadata-index"
USER_INDEX_NAME = "user-preferences-index"

# Pinecone clients (ensure singleton/efficient usage in real app)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
book_index = pc.Index(BOOK_INDEX_NAME)
user_index = pc.Index(USER_INDEX_NAME)

async def dense_vector_recommendation(user_id: str, top_k: int = 50) -> Tuple[List[str], List[float]]:
    # 1. Get user vector
    query_result = user_index.fetch(ids=[str(user_id)], namespace="__default__")
    user_vectors = query_result.vectors
    user_record = user_vectors.get(str(user_id), None)
    user_vector = user_record.values if user_record else None
    if not user_vector:
        return [], []

    # 2. Query books using dense user vector
    results = book_index.query(vector=user_vector, top_k=top_k, namespace="__default__")

    # 3. Extract book ids and scores (similarity/distance depending on Pinecone metric)
    book_ids = []
    scores = []
    for match in results.matches:
        book_id = match.get("_id") or match.get("id")
        score = match.get("score")
        if book_id is not None and score is not None:
            book_ids.append(book_id)
            scores.append(score)
    print(f"1. CB Recommendations for user {user_id}: {book_ids} with scores {scores} with top_k {top_k}")
    return book_ids, scores
