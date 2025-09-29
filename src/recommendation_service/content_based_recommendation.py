import os
import httpx
from pinecone import Pinecone
from typing import List

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BOOK_INDEX_NAME = "book-metadata-index"
USER_INDEX_NAME = "user-preferences-index"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

# Pinecone clients (ensure singleton/efficient usage in real app)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
book_index = pc.Index(BOOK_INDEX_NAME)
user_index = pc.Index(USER_INDEX_NAME)

async def get_books_by_ids(ids: List[str]):
    if not ids:
        return []
    async with httpx.AsyncClient() as client:
        idlist = ",".join([f'"{bid}"' for bid in ids])
        fields = "id,title,authors,categories,thumbnail_url,download_link"
        url = f"{SUPABASE_URL}/rest/v1/books?select={fields}&id=in.({idlist})"
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def dense_vector_recommendation(user_id: str, top_k: int = 50):
    # 1. Get user vector
    query_result = user_index.fetch(ids=[str(user_id)], namespace="__default__")
    user_vectors = query_result.vectors
    user_record = user_vectors.get(str(user_id), None)
    user_vector = user_record.values if user_record else None
    if not user_vector:
        return {"recommendations": []}

    # 2. Query books using dense user vector
    results = book_index.query(vector=user_vector, top_k=top_k, namespace="__default__")
    book_ids = [match.get("_id") or match.get("id") for match in results.matches if match.get("_id") or match.get("id")]

    # 3. Get metadata for recommended books
    books = await get_books_by_ids(book_ids)
    return {"recommendations": books}
