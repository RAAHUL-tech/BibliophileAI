from datetime import datetime 
import os
import redis
import httpx
import jwt
import random
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, Depends, Security, HTTPException, Query, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import content_based_recommendation as cbr
import collaborative_filtering as cf
from graph_recommendation import graph_recommend_books
from sasrec_inference import recommend_for_session
from popularity_recommendation import get_popularity_recommendations, get_s3_popularity_fallback
from linucb_inference import linucb_ranker
from feature_engineering.feature_service import feature_service

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
combined_scores_map = defaultdict(dict)

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
            "popularity_scores": scores,
            "source": source
        }
    }


@app.get("/api/v1/recommend/combined")
async def recommend_combined(
    current_user: dict = Depends(get_current_user),
    request: Request = None
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
    session_book_ids, session_scores = recommend_for_session(user_id, top_k=100)
    
    # Popularity-based (NEW: calls popularity service endpoint)
    pop_book_ids, pop_scores = await get_popularity_recommendations(user_id, top_k=100)
    print(f"Got {len(cb_book_ids)} CB, {len(als_book_ids)} ALS, {len(graph_book_ids)} Graph, "
          f"{len(session_book_ids)} Session, {len(pop_book_ids)} Popularity books")
    
    all_candidates = list(set(
        cb_book_ids + als_book_ids + graph_book_ids + 
        session_book_ids + pop_book_ids
    ))
    print(f"Combined {len(all_candidates)} unique candidates from 5 sources")

    linucb_book_ids, linucb_scores = linucb_ranker.get_linucb_ranked(all_candidates, user_id)

    # 2. Take equal share from each source
    target_total = 1000
    num_sources = 6  # CB + ALS + Graph + Session + Popularity + LinUCB
    target_each = max(1, target_total // num_sources)
    
    cb_final = cb_book_ids[:target_each]
    cb_final_scores = cb_scores[:target_each]
    
    als_final = als_book_ids[:target_each]
    als_final_scores = cf_scores[:target_each]
    
    graph_final = graph_book_ids[:target_each]
    graph_final_scores = graph_scores[:target_each]
    
    session_final = session_book_ids[:target_each]
    session_final_scores = session_scores[:target_each]

    pop_final = pop_book_ids[:target_each]
    pop_final_scores = pop_scores[:target_each]

    linucb_final = linucb_book_ids[:target_each]
    linucb_final_scores = linucb_scores[:target_each]

    # 3. Combine all
    add_scores(cb_final, cb_final_scores, "content")
    add_scores(als_final, als_final_scores, "als")
    add_scores(graph_final, graph_final_scores, "graph")
    add_scores(session_final, session_final_scores, "session")
    add_scores(pop_final, pop_final_scores, "popularity")
    add_scores(linucb_final, linucb_final_scores, "linucb")
    print(f"Combined unique book IDs before shuffling: {len(combined_scores_map)}")

    all_book_ids = list(combined_scores_map.keys())
    session_context = await build_session_context(current_user, request)
    features = await feature_service.engineer_features_batch(
        user_id, all_book_ids, combined_scores_map, session_context
    )
    print(f"Engineered features for {len(features)} books")
    print("Features:", features)
    # 4. Shuffle for serendipity
    all_book_ids = list(combined_scores_map.keys())
    random.shuffle(all_book_ids)

    final_ids = all_book_ids[:target_total]
    
    # 6. Fetch metadata and attach scores
    books = await get_books_by_ids(list(final_ids))
    
    recommendations = []
    
    for book in books:
        bid = book.get("id") or book.get("_id")
        if bid in combined_scores_map:
            book["scores"] = combined_scores_map[bid]
            book["source"] = list(combined_scores_map[bid].keys())
        recommendations.append(book)
    
    print(f"Returning {len(recommendations)} final recommendations")
    return {"recommendations": recommendations}


def add_scores(book_ids, scores, source_name):
    for bid, score in zip(book_ids, scores):
        combined_scores_map[bid][f"{source_name}_score"] = float(score)

async def build_session_context(current_user: dict, request: Request = None) -> Dict[str, Any]:
    """
    Build complete session context for feature engineering
    Used in: feature_service.engineer_features_batch()
    """
    user_id = str(current_user["id"])
    session_context = {
        # Device detection (from headers/user-agent)
        "device": _detect_device(request),
        
        # Current hour (time-based features)
        "hour_of_day": datetime.now().hour,
        
        # User agent details
        "user_agent": request.headers.get("user-agent", "") if request else "",
    }
    
    logger.info(f"Session context for {user_id}: {session_context}")
    return session_context


def _detect_device(request: Request = None) -> str:
    """Detect device type from User-Agent"""
    if not request:
        return "desktop"
    
    ua = request.headers.get("user-agent", "").lower()
    if any(x in ua for x in ["mobile", "android", "iphone", "ipad"]):
        return "mobile"
    elif "tablet" in ua or "ipad" in ua:
        return "tablet"
    return "desktop"
