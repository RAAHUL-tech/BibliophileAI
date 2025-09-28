import os
import asyncio
import httpx
from bs4 import BeautifulSoup

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

PAGE_SIZE = 1000

async def fetch_books(limit, offset):
    url = f"{SUPABASE_URL}/rest/v1/books?select=id,info_link&limit={limit}&offset={offset}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

async def fetch_page_html(url):
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text
    
def extract_description(html):
    soup = BeautifulSoup(html, "html.parser")
    readmore = soup.find("span", class_="readmore-container")
    toggle = soup.find("span", class_="toggle-content")
    summary = readmore.get_text(strip=True) if readmore else ""
    extra = toggle.get_text(strip=True) if toggle else ""
    return summary + " " + extra

async def update_description(book_id, description):
    url = f"{SUPABASE_URL}/rest/v1/books?id=eq.{book_id}"
    json_data = {'description': description}
    async with httpx.AsyncClient() as client:
        response = await client.patch(url, json=json_data, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return None

async def main():
    offset = 0
    while True:
        books = await fetch_books(PAGE_SIZE, offset)
        if not books:
            break
        tasks = []
        for book in books:
            if not book.get('info_link'):
                continue
            async def task(bk):
                try:
                    html = await fetch_page_html(bk['info_link'])
                    desc = extract_description(html)
                    print(desc)
                    if desc:
                        await update_description(bk['id'], desc)
                        print(f"Updated description for book id {bk['id']}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 504:
                       print(f"Timeout fetching {bk['info_link']} for book id {bk['id']}, setting description null")
                       await update_description(bk['id'], None)
                    else:
                        print(f"HTTP error updating book id {bk['id']}: {e}")
                    print(f"Error updating book id {bk['id']}: {e}")
                except Exception as e:
                   print(f"Error updating book id {bk['id']}: {e}")
            tasks.append(task(book))
        await asyncio.gather(*tasks)
        offset += PAGE_SIZE


if __name__ == '__main__':
    asyncio.run(main())
