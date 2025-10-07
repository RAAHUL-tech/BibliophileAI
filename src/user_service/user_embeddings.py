import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "user-preferences-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

async def create_user_embedding_vectors(user_id: str, genres: list, authors: list, age: int, pincode: str):
    genres_text = f"genres: {', '.join(genres)}" if genres else "genres: none"
    authors_text = f"authors: {', '.join(authors)}" if authors else "authors: none"
    age_text = f"age: {age}" if age else "age: unknown"
    pincode_text = f"pincode: {pincode}" if pincode else "pincode: unknown"
    
    # Combine all preference data into a single text string
    combined_text = f"{genres_text}, {authors_text}, {age_text}, {pincode_text}"
    record = {
        "_id": str(user_id),  
        "text": combined_text
    }
    try:
        index.upsert_records(namespace="__default__", records=[record])
    except Exception as e:
        print(f"Upsert failed for user_id {user_id}: {e}")
