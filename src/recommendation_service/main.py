import os
import redis
import httpx
import jwt
import random
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, Depends, Security, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

import content_based_recommendation as cbr
import collaborative_filtering as cf
from graph_recommendation import graph_recommend_books
from sasrec_inference import recommend_for_session
from popularity_recommendation import get_popularity_recommendations, get_s3_popularity_fallback

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"


POPULARITY_WINDOWS = ["multi", "7d", "30d", "90d"]
POPULARITY_KEY_PATTERN = "popularity:trending:{window}"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.from_url(os.environ["REDIS_URL"])


def decode_access_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def get_user_by_username(username: str) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        params = {"username": f"eq.{username}"}
        url = f"{SUPABASE_URL}/rest/v1/users"
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        users = response.json()
        return users[0] if users else None

async def get_current_user(token: str = Security(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = decode_access_token(token)
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    user = await get_user_by_username(username)
    if not user:
        raise credentials_exception
    return user

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

@app.get("/api/v1/recommend/graph")
async def recommend_graph(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["id"])
    loop = asyncio.get_running_loop()
    # graph_recommend_books is sync; run in thread
    graph_results = await loop.run_in_executor(None, graph_recommend_books, user_id, 50)
    # graph_results: List[(book_id, graph_score)]
    book_ids = [bid for bid, _ in graph_results]
    scores = [score for _, score in graph_results]
    books = await get_books_by_ids(book_ids)
    score_map = dict(zip(book_ids, scores))
    for b in books:
        bid = b.get("id") or b.get("_id")
        if bid in score_map:
            b["graph_score"] = score_map[bid]
    return {"recommendations": books}

@app.get("/api/v1/recommend/session")
async def recommend_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    book_ids = recommend_for_session(user_id, top_k=20)
    if not book_ids:
        return {"recommendations": []}
    books = await get_books_by_ids(book_ids)
    return {"recommendations": books}

@app.get("/api/v1/recommend/popularity/global")
async def get_global_popularity_recommendations(
    window: str = Query("multi", description="Popularity window"),
    top_k: int = Query(50, ge=1, le=100, description="Number of recommendations")
):
    """
    Get global top trending books (Redis primary + S3 PyTorch fallback).
    Returns: {"recommendations": {"book_ids": [...], "popularity_scores": [...]}}
    """
    if window not in POPULARITY_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Invalid window. Choose: {POPULARITY_WINDOWS}")
    
    key = POPULARITY_KEY_PATTERN.format(window=window)
    
    # Try Redis first (primary source)
    try:
        if redis_client.exists(key) and redis_client.zcard(key) > 0:
            books_scores = redis_client.zrevrange(key, 0, top_k - 1, withscores=True)
            if books_scores:
                book_ids = [book_id_bytes.decode("utf-8") for book_id_bytes, _ in books_scores]
                scores = [float(score) for _, score in books_scores]
                source = "redis"
                print(f"Redis hit: {len(book_ids)} global books for {window}")
            else:
                book_ids, scores, source = [], [], "empty_redis"
        else:
            book_ids, scores, source = [], [], "no_redis_key"
    except Exception as e:
        print(f"Redis failed: {e}")
        book_ids, scores, source = [], [], "redis_error"
    
    # Fallback to S3 if Redis unavailable or empty
    if not book_ids:
        book_ids, scores = get_s3_popularity_fallback(window, top_k)
        source = "s3_fallback"
        print(f"S3 fallback: {len(book_ids)} global books for {window}")
    
    return {
        "recommendations": {
            "book_ids": book_ids,
            "popularity_scores": scores
        },
        "window": window,
        "total_books": len(book_ids),
        "source": source
    }


@app.get("/api/v1/recommend/combined")
async def recommend_combined(
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["id"])

    # 1. Get IDs and relevance scores from ALL recommenders (5 total)
    print(f"Generating recommendations for user {user_id}")
    
    # Content-based
    cb_book_ids, cb_scores = await cbr.dense_vector_recommendation(user_id, top_k=100)
    
    # Collaborative filtering (ALS)
    als_book_ids, cf_scores = await cf.recommend_als_books(user_id, top_k=100)
    
    # Graph-based
    loop = asyncio.get_running_loop()
    graph_results = await loop.run_in_executor(None, graph_recommend_books, user_id, 100)
    graph_book_ids = [bid for bid, _ in graph_results]
    graph_scores = [score for _, score in graph_results]
    
    # Session-based (SASRec)
    session_book_ids = recommend_for_session(user_id, top_k=100)
    session_scores = [1.0] * len(session_book_ids)
    
    # Popularity-based (NEW: calls popularity service endpoint)
    pop_book_ids, pop_scores = await get_popularity_recommendations(user_id, top_k=100)
    print(f"Got {len(cb_book_ids)} CB, {len(als_book_ids)} ALS, {len(graph_book_ids)} Graph, "
          f"{len(session_book_ids)} Session, {len(pop_book_ids)} Popularity books")

    # 2. Deduplicate: content-based as base set
    cb_set = set(cb_book_ids)
    
    # Filter unique from other sources
    als_unique, als_unique_scores = filter_unique_books(als_book_ids, cf_scores, cb_set)
    graph_unique, graph_unique_scores = filter_unique_books(graph_book_ids, graph_scores, cb_set | set(als_unique))
    session_unique, session_unique_scores = filter_unique_books(session_book_ids, session_scores, 
                                                               cb_set | set(als_unique) | set(graph_unique))
    pop_unique, pop_unique_scores = filter_unique_books(pop_book_ids, pop_scores,
                                                       cb_set | set(als_unique) | set(graph_unique) | set(session_unique))
    
    # 3. Take equal share from each source (5 sources now)
    target_total = 50
    num_sources = 5  # CB + ALS + Graph + Session + Popularity
    target_each = max(1, target_total // num_sources)
    
    cb_final = cb_book_ids[:target_each]
    cb_final_scores = cb_scores[:target_each]
    
    als_final = als_unique[:target_each]
    als_final_scores = als_unique_scores[:target_each]
    
    graph_final = graph_unique[:target_each]
    graph_final_scores = graph_unique_scores[:target_each]
    
    session_final = session_unique[:target_each]
    session_final_scores = session_unique_scores[:target_each]
    
    pop_final = pop_unique[:target_each]
    pop_final_scores = pop_unique_scores[:target_each]
    
    # 4. Top up if < target_total
    all_remaining = (als_unique[target_each:] + graph_unique[target_each:] + 
                    session_unique[target_each:] + pop_unique[target_each:])
    all_remaining_scores = (als_unique_scores[target_each:] + graph_unique_scores[target_each:] + 
                           session_unique_scores[target_each:] + pop_unique_scores[target_each:])
    
    combined_ids = cb_final + als_final + graph_final + session_final + pop_final
    combined_scores = cb_final_scores + als_final_scores + graph_final_scores + session_final_scores + pop_final_scores
    
    # Fill remaining slots
    for bid, score in zip(all_remaining, all_remaining_scores):
        if len(combined_ids) >= target_total:
            break
        if bid not in combined_ids:
            combined_ids.append(bid)
            combined_scores.append(score)
    
    # 5. Shuffle for serendipity
    paired = list(zip(combined_ids, combined_scores))
    random.shuffle(paired)
    final_ids, final_scores = zip(*paired[:target_total]) if paired else ([], [])
    
    # 6. Fetch metadata and attach scores
    books = await get_books_by_ids(list(final_ids))
    score_map = dict(zip(final_ids, final_scores))
    
    recommendations = []
    source_labels = {0: "content", 1: "als", 2: "graph", 3: "session", 4: "popularity"}
    
    for book in books:
        bid = book.get("id") or book.get("_id")
        if bid in score_map:
            book["combined_score"] = float(score_map[bid])
            book["source"] = "combined"
        recommendations.append(book)
    
    print(f"Returning {len(recommendations)} final recommendations")
    return {"recommendations": recommendations}



def filter_unique_books(book_ids: List[str], scores: List[float], existing_set: set) -> tuple[List[str], List[float]]:
    """Filter books not already in existing set"""
    unique_ids, unique_scores = [], []
    for bid, score in zip(book_ids, scores):
        if bid not in existing_set:
            unique_ids.append(bid)
            unique_scores.append(score)
    return unique_ids, unique_scores
