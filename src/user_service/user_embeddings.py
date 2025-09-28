import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "user-preferences-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

async def create_user_embedding_vectors(user_id: str, genres: list):
    combined_text = ", ".join(genres) if genres else ""
    record = {
        "_id": str(user_id),  
        "text": combined_text
    }
    try:
        index.upsert_records(namespace="__default__", records=[record])
    except Exception as e:
        print(f"Upsert failed for user_id {user_id}: {e}")
