import os
import asyncio
import httpx
from neo4j import GraphDatabase

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")  
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
PAGE_SIZE = 1000

# Prepare Supabase headers
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

async def fetch_books(limit, offset):
    url = f"{SUPABASE_URL}/rest/v1/books?select=id,title,authors,categories,language,download_link&limit={limit}&offset={offset}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

async def get_all_books(total_records):
    books = []
    tasks = []
    for offset in range(0, total_records, PAGE_SIZE):
        tasks.append(fetch_books(PAGE_SIZE, offset))
    results = await asyncio.gather(*tasks)
    for result in results:
        books.extend(result)
    return books

def store_books_to_neo4j(driver, books):
    with driver.session() as session:
        for book in books:
            # Store the Book node
            session.run(
                """
                MERGE (b:Book {id: $id})
                SET b.title = $title, b.language = $language, b.download_link = $download_link
                """,
                id=book.get('id'),
                title=book.get('title'),
                language=book.get('language'),
                download_link=book.get('download_link')
            )
            # Store Author nodes and WROTE relationship
            for author in book.get('authors', []):
                session.run(
                    """
                    MERGE (a:Author {name: $author})
                    MERGE (b:Book {id: $book_id})
                    MERGE (a)-[:WROTE]->(b)
                    """,
                    author=author,
                    book_id=book.get('id')
                )
            # Store Genre nodes and HAS GENRE relationship
            for genre in book.get('categories', []):
                session.run(
                    """
                    MERGE (g:Genre {name: $genre})
                    MERGE (b:Book {id: $book_id})
                    MERGE (b)-[:HAS_GENRE]->(g)
                    """,
                    genre=genre,
                    book_id=book.get('id')
                )

async def main():
    # Step 1: Fetch all books
    total_records = 3364  # Your collection size
    books = await get_all_books(total_records)
    print(f"Fetched {len(books)} books from Supabase.")
    # Step 2: Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j database.")
    # Step 3: Store data in Neo4j
    store_books_to_neo4j(driver, books)
    print("Stored books in Neo4j database.")
    # Step 4: Close the driver
    driver.close()

# To run the script
# asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())