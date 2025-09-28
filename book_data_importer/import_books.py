import asyncio
import httpx
import os
import random

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates"
}

genres = ["Fantasy",
  "Romance",
  "Science Fiction",
  "Mystery",
  "Thriller",
  "Non-fiction",
  "Historical",
  "Young Adult",
  "Horror",
  "Biography"]

async def fetch_books_by_genre(genre, page=1, page_size=40):
    url = "https://gutendex.com/books/"
    params = {
        "subject": genre.lower(),
        "page": page,
        "page_size": page_size
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Fetch error: {repr(e)}")
            return None


def parse_book(item):
    return {
        "id": str(item.get("id")),
        "title": item.get("title", ""),
        "authors": [author.get("name") for author in item.get("authors", [])],
        "categories": item.get("subjects", []),
        "language": item.get("languages", [None])[0],
        "thumbnail_url": item.get("formats", {}).get("image/jpeg"),
        "info_link": f"https://www.gutenberg.org/ebooks/{item.get('id')}",
        "epub_available": "application/epub+zip" in item.get("formats", {}),
        "pdf_available": "application/pdf" in item.get("formats", {}),
        "download_link": item.get("formats", {}).get("application/pdf")
            or item.get("formats", {}).get("application/epub+zip"),
    }

async def store_book(book):
    url = f"{SUPABASE_URL}/rest/v1/books"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=book, headers=headers)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None

async def main():
    stored_books = 0
    page = 1
    while stored_books < 4000:  # adjust as needed
        genre = random.choice(genres)
        try:
            data = await fetch_books_by_genre(genre, page)
            if not data:
                await asyncio.sleep(5)
                continue
        except Exception as e:
            print(f"Fetch error: {repr(e)}")
            await asyncio.sleep(5)
            continue
        items = data.get("results", [])
        for item in items:
            book = parse_book(item)
            if not (book["epub_available"] or book["pdf_available"]):
                continue
            await store_book(book)
            stored_books += 1
            if stored_books >= 1000:
                break
            await asyncio.sleep(0.2)
        page += 1

if __name__ == "__main__":
    asyncio.run(main())
