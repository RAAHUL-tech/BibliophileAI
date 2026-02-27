import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pinecone import Pinecone


logger = logging.getLogger("search_service")
logging.basicConfig(level=logging.INFO)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BOOK_INDEX_NAME = "book-metadata-index"
PINECONE_EMBEDDING_MODEL = "llama-text-embed-v2"
BOOKS_FETCH_TIMEOUT = float(os.getenv("BOOKS_FETCH_TIMEOUT", "60.0"))

supabase_headers = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}


app = FastAPI(title="BibliophileAI Search Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pc: Optional[Pinecone] = None
book_index = None

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        book_index = pc.Index(BOOK_INDEX_NAME)
        logger.info("Connected to Pinecone index %s", BOOK_INDEX_NAME)
    except Exception as e:
        logger.error("Failed to initialise Pinecone: %s", e)
else:
    logger.warning("PINECONE_API_KEY not set; semantic search will be disabled.")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text search query")
    top_k: int = Field(20, ge=1, le=100, description="Maximum number of books to return")


class Book(BaseModel):
    id: str
    title: str
    authors: List[str] = []
    categories: List[str] = []
    thumbnail_url: Optional[str] = None
    download_link: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[Book]


async def get_books_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch book details from Supabase books table by list of IDs.
    This is the source of truth for book metadata; Pinecone only stores vector IDs.
    """
    if not ids:
        return []
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set; cannot fetch books.")
        return []

    try:
        async with httpx.AsyncClient(timeout=BOOKS_FETCH_TIMEOUT) as client:
            idlist = ",".join([f'"{bid}"' for bid in ids])
            fields = "id,title,authors,categories,thumbnail_url,download_link,language"
            url = f"{SUPABASE_URL}/rest/v1/books?select={fields}&id=in.({idlist})"
            resp = await client.get(url, headers=supabase_headers)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
        logger.warning(f"Search service: books fetch failed (timeout/connect): {e}. Check SUPABASE_URL and network.")
        return []
    except Exception as e:
        logger.warning(f"Search service: books fetch failed: {e}")
        return []


def _get_query_embedding(text: str) -> List[float]:
    """
    Get a vector embedding for the query text using Pinecone's
    built-in inference API (if available).
    """
    if not pc:
        raise RuntimeError("Pinecone client not initialised")

    try:
        # Use Pinecone Inference embed API with required input_type parameter.
        res = pc.inference.embed(
            model=PINECONE_EMBEDDING_MODEL,
            parameters={"input_type": "query", "truncate": "END"},
            inputs=[{"text": text}],
        )
        # Response shape: {"data": [{"values": [...], ...}], ...}
        data = getattr(res, "data", None) or res.get("data", [])
        if not data:
            raise RuntimeError("No embedding data returned from Pinecone")
        first = data[0]
        values = getattr(first, "values", None) or first.get("values")
        if values is None:
            raise RuntimeError("No embedding values in Pinecone response")
        vec = list(values) if not isinstance(values, list) else values
        return vec
    except Exception as e:
        logger.error("Failed to embed query with Pinecone: %s", e)
        raise RuntimeError("Embedding failed") from e


@app.post("/api/v1/search/", response_model=SearchResponse)
async def search_books(payload: SearchRequest) -> SearchResponse:
    """
    Semantic search over books.

    1. Embed the query text using Pinecone.
    2. Query the book index for the most similar books.
    3. Return book metadata for the frontend to display.
    """
    print(f"Search request: {payload.query}")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if not book_index:
        raise HTTPException(status_code=503, detail="Search index not available.")

    try:
        query_vec = _get_query_embedding(payload.query.strip())
        print(f"Query vector: {query_vec}")
    except RuntimeError as e:
        logger.error("Search failed while embedding: %s", e)
        raise HTTPException(status_code=503, detail="Search embedding failed.") from e

    try:
        res = book_index.query(
            vector=query_vec,
            top_k=payload.top_k,
            include_metadata=True,
        )
        print(f"Pinecone query result: {res}")
    except Exception as e:
        logger.error("Pinecone query failed: %s", e)
        raise HTTPException(status_code=503, detail="Search backend error.") from e

    matches = getattr(res, "matches", None) or res.get("matches", [])
    print(f"Pinecone matches: {matches}")
    # Extract IDs and scores from Pinecone matches
    book_ids: List[str] = []
    scores: List[Optional[float]] = []
    for m in matches:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("_id")
            score = m.get("score")
        else:
            mid = getattr(m, "id", None) or getattr(m, "_id", None)
            score = getattr(m, "score", None)
        if mid is None:
            continue
        book_ids.append(str(mid))
        scores.append(float(score) if score is not None else None)

    # Fetch canonical book metadata from Supabase
    books_raw = await get_books_by_ids(book_ids)
    books_map: Dict[str, Dict[str, Any]] = {
        str(b.get("id") or b.get("_id")): b for b in books_raw
    }

    results: List[Book] = []
    for bid, sc in zip(book_ids, scores):
        raw = books_map.get(bid)
        if not raw:
            continue
        authors_raw = raw.get("authors") or []
        if isinstance(authors_raw, str):
            authors = [authors_raw]
        else:
            authors = list(authors_raw)

        categories_raw = raw.get("categories") or []
        if isinstance(categories_raw, str):
            categories = [categories_raw]
        else:
            categories = list(categories_raw)

        results.append(
            Book(
                id=str(raw.get("id") or bid),
                title=raw.get("title") or "",
                authors=authors,
                categories=categories,
                thumbnail_url=raw.get("thumbnail_url"),
                download_link=raw.get("download_link"),
                score=sc,
            )
        )

    return SearchResponse(results=results)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

