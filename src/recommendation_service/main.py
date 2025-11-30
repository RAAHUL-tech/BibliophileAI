import os
import httpx
import jwt
import random
import asyncio
from typing import Optional, List

from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

import content_based_recommendation as cbr
import collaborative_filtering as cf
from graph_recommendation import graph_recommend_books

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

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

@app.get("/api/v1/recommend/combined")
async def recommend_combined(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["id"])

    # 1. Get IDs and relevance scores from three recommenders
    cb_book_ids, cb_scores = await cbr.dense_vector_recommendation(user_id, top_k=100)
    als_book_ids, cf_scores = await cf.recommend_als_books(user_id, top_k=100)

    loop = asyncio.get_running_loop()
    graph_results = await loop.run_in_executor(None, graph_recommend_books, user_id, 100)
    graph_book_ids = [bid for bid, _ in graph_results]
    graph_scores = [score for _, score in graph_results]
    print(f"Graph Recommendations for user {user_id}: {graph_book_ids} with scores {graph_scores}")
    # 2. Deduplicate (content-based as base set)
    cb_set = set(cb_book_ids)

    als_unique = []
    als_unique_scores = []
    for bid, s in zip(als_book_ids, cf_scores):
        if bid not in cb_set:
            als_unique.append(bid)
            als_unique_scores.append(s)

    graph_unique = []
    graph_unique_scores = []
    existing = cb_set.union(als_unique)
    for bid, s in zip(graph_book_ids, graph_scores):
        if bid not in existing:
            graph_unique.append(bid)
            graph_unique_scores.append(s)

    # 3. Take roughly 1/3 from each source (up to available)
    target_each = 50 // 3 or 1

    cb_final = cb_book_ids[:target_each]
    cb_final_scores = cb_scores[:target_each]

    als_final = als_unique[:target_each]
    als_final_scores = als_unique_scores[:target_each]

    graph_final = graph_unique[:target_each]
    graph_final_scores = graph_unique_scores[:target_each]

    # If total < 50, top up from remaining pools
    all_ids_pool = cb_book_ids + als_unique + graph_unique
    all_scores_pool = cb_scores + als_unique_scores + graph_unique_scores

    chosen_ids = set(cb_final + als_final + graph_final)
    combined_ids = cb_final + als_final + graph_final
    combined_scores = cb_final_scores + als_final_scores + graph_final_scores

    for bid, s in zip(all_ids_pool, all_scores_pool):
        if len(combined_ids) >= 50:
            break
        if bid not in chosen_ids:
            combined_ids.append(bid)
            combined_scores.append(s)
            chosen_ids.add(bid)

    # 4. Shuffle final list
    paired = list(zip(combined_ids, combined_scores))
    random.shuffle(paired)
    paired = paired[:50]
    shuffled_ids, shuffled_scores = zip(*paired) if paired else ([], [])

    # 5. Fetch metadata and attach combined_score
    books = await get_books_by_ids(list(shuffled_ids))
    score_map = dict(zip(shuffled_ids, shuffled_scores))

    recommendations = []
    for book in books:
        bid = book.get("id") or book.get("_id")
        if bid in score_map:
            book["combined_score"] = score_map[bid]
        recommendations.append(book)

    return {"recommendations": recommendations}
