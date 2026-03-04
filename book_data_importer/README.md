# Book Data Importer

Scripts and utilities to populate the book catalog and related data used by search and recommendations: Supabase books table, Pinecone book embeddings, Neo4j graph, and S3 EPUBs.

## Functionality

| Script | Purpose |
|--------|--------|
| **import_books.py** | Fetches books from [Gutendex](https://gutendex.com/) by genre (Fantasy, Romance, Sci-Fi, etc.), normalizes metadata (title, authors, categories, language, download_link), and upserts into the Supabase `books` table. |
| **book_embedding.py** | Reads all books from Supabase, computes or fetches embeddings (e.g. via Pinecone or an embed API), and upserts vectors into the Pinecone `book-metadata-index` for semantic search and content-based recommendation. |
| **graph_book_importer.py** | Reads books from Supabase and creates Neo4j nodes (Book, Author, Genre) and relationships (WROTE, HAS_GENRE, etc.) so the graph-based recommendation and graph training job can use the catalog. |
| **scrape_descriptions.py** | For each book in Supabase (with `info_link`), fetches the page, extracts description text (e.g. via BeautifulSoup), and updates the book row with a description field. |
| **upload_epubs_to_s3.py** | Fetches book records from Supabase (with `download_link`), downloads each EPUB, and uploads to AWS S3 under a configurable prefix (e.g. `books-epub/`) for the in-app reader. |

## Implementation in this project

- **Data flow**: Gutendex → import_books → Supabase. Supabase is the source of truth for book metadata. Then book_embedding → Pinecone; graph_book_importer → Neo4j; scrape_descriptions → Supabase; upload_epubs_to_s3 → S3.
- **Environment**: All scripts use `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`; Pinecone/Neo4j/S3 as needed per script. See each script’s docstring and env usage.
- **Run order**: Typically (1) import_books, (2) scrape_descriptions (optional), (3) book_embedding, (4) graph_book_importer, (5) upload_epubs_to_s3. Can be run manually or wired into a one-off/CI pipeline.

## Dependencies

See `requirements.txt` in this folder (httpx, pinecone, neo4j, beautifulsoup4, boto3, supabase, etc.).
