import os
import asyncio
import httpx
from pinecone import Pinecone
from itertools import islice


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1-aws"
INDEX_NAME = "book-metadata-index"
BATCH_SIZE = 96  


headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


async def fetch_all_books():
    all_books = []
    page = 0
    limit = 1000
    while True:
        url = f"{SUPABASE_URL}/rest/v1/books?select=id,title,authors,categories,language&limit={limit}&offset={page*limit}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            books = resp.json()
            if not books:
                break
            all_books.extend(books)
            page += 1
    return all_books


def chunked(iterable, n):
    iterator = iter(iterable)
    for first in iterator:
        batch = [first] + list(islice(iterator, n - 1))
        yield batch


async def batched_upsert(books):
    records = []
    for book in books:
        text = f"{book.get('title','')}. {', '.join(book.get('authors', []))}. {', '.join(book.get('categories', []))}."
        records.append({
            "_id": book["id"],
            "text": text,
            "language": book.get("language")  
        })
    for batch in chunked(records, BATCH_SIZE):
        index.upsert_records("__default__", batch)
        print(f"Upserted batch of {len(batch)} books to Pinecone.")
        await asyncio.sleep(10)


async def main():
    books = await fetch_all_books()
    print(f"Fetched total {len(books)} books from Supabase")
    await batched_upsert(books)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
