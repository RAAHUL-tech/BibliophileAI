from datetime import datetime, timedelta
import os
import json
import redis
import httpx
import jwt
import random
import asyncio
import logging
from pymongo import MongoClient
from feast import FeatureStore
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, Depends, Security, HTTPException, Query, Request, Header
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import content_based_recommendation as cbr
from content_based_recommendation import book_index as pinecone_book_index
import collaborative_filtering as cf
import numpy as np 
from graph_recommendation import graph_recommend_books
from sasrec_inference import recommend_for_session
from popularity_recommendation import get_popularity_recommendations, get_s3_popularity_fallback
from linucb_inference import linucb_ranker
from feature_engineering.feature_service import feature_service, get_feast_repo_path
from ltr_ranking import rank_candidates as ltr_rank_candidates


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

# Internal API key for background pods (set INTERNAL_API_KEY env)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "bibliophileai")


def verify_internal_key(x_internal_key: Optional[str] = Header(None, alias="X-Internal-Key")):
    if not INTERNAL_API_KEY or x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


class RefreshBody(BaseModel):
    user_id: str


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

# Per-user recommendation cache: always serve from cache; background pods update on events.
RECOMMEND_CACHE_KEY_PREFIX = "recommend:combined:"
RECOMMEND_CACHE_TTL_SECONDS = 365 * 24 * 3600  # 1 year


def _cache_key(user_id: str) -> str:
    return f"{RECOMMEND_CACHE_KEY_PREFIX}{user_id}"


def get_cached_recommendations(user_id: str) -> Optional[Dict[str, Any]]:
    """Return cached combined recommendations (by category) or None if miss. Excludes Trending Now (filled from global Redis)."""
    try:
        raw = redis_client.get(_cache_key(user_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache get failed for user {user_id}: {e}")
        return None

def set_cached_recommendations(user_id: str, payload: Dict[str, Any]) -> None:
    """Store full combined recommendations and set 1-year TTL."""
    try:
        key = _cache_key(user_id)
        redis_client.setex(key, RECOMMEND_CACHE_TTL_SECONDS, json.dumps(payload))
        logger.info(f"Cache set for user {user_id}, {len(payload.get('categories', []))} categories")
    except Exception as e:
        logger.warning(f"Cache set failed for user {user_id}: {e}")


def update_cached_category(
    user_id: str,
    category_name: str,
    category_payload: Dict[str, Any],
) -> None:
    """
    Update one category in the user's cached recommendations (used by background pods).
    category_payload: {"category": str, "description": str, "books": [...]}
    If cache miss, creates a cache with only this category.
    """
    try:
        key = _cache_key(user_id)
        raw = redis_client.get(key)
        if raw:
            payload = json.loads(raw)
            categories = list(payload.get("categories") or [])
        else:
            categories = []
        # Replace or append category by name
        categories = [c for c in categories if c.get("category") != category_name]
        categories.append(category_payload)
        payload = {"categories": categories}
        redis_client.setex(key, RECOMMEND_CACHE_TTL_SECONDS, json.dumps(payload))
        logger.info(f"Cache updated category '{category_name}' for user {user_id}")
    except Exception as e:
        logger.warning(f"Cache update category failed for user {user_id}: {e}")


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

# Supabase/books request timeout (avoid 500 on slow or unreachable Supabase)
BOOKS_FETCH_TIMEOUT = float(os.getenv("BOOKS_FETCH_TIMEOUT", "60.0"))

async def get_books_by_ids(ids: List[str]):
    if not ids:
        return []
    try:
        async with httpx.AsyncClient(timeout=BOOKS_FETCH_TIMEOUT) as client:
            idlist = ",".join([f'"{bid}"' for bid in ids])
            fields = "id,title,authors,categories,thumbnail_url,download_link"
            url = f"{SUPABASE_URL}/rest/v1/books?select={fields}&id=in.({idlist})"
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
        logger.warning(f"Books fetch failed (timeout/connect): {e}. Check SUPABASE_URL and network.")
        return []
    except Exception as e:
        logger.warning(f"Books fetch failed: {e}")
        return []


async def get_user_preferred_genres(user_id: str) -> List[str]:
    """Fetch user's preferred genres from Supabase user_preferences."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{SUPABASE_URL}/rest/v1/user_preferences"
            params = {"user_id": f"eq.{user_id}", "select": "genres"}
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            rows = resp.json()
            if not rows or not isinstance(rows, list):
                return []
            genres = rows[0].get("genres") if rows else []
            return list(genres) if isinstance(genres, (list, tuple)) else []
    except Exception as e:
        logger.warning(f"Failed to fetch user preferences: {e}")
        return []

async def _get_trending_now_category(top_k: int = 50) -> Dict[str, Any]:
    """
    Build 'Trending Now' category from global Redis key (popularity:trending:multi)
    populated by model_training_service/popularity_train. Same for all users; not stored per user.
    """
    pop_book_ids, pop_scores = await get_popularity_recommendations("", window="multi", top_k=top_k)
    if not pop_book_ids:
        return {"category": "Trending Now", "description": "Popular books right now", "books": []}
    books_list = await get_books_by_ids(pop_book_ids)
    books_dict = {str(b.get("id") or b.get("_id")): b for b in books_list}
    books = []
    for bid, score in zip(pop_book_ids, pop_scores):
        if bid in books_dict:
            b = books_dict[bid].copy()
            b["score"] = float(score)
            books.append(b)
    return {"category": "Trending Now", "description": "Popular books right now", "books": books}



async def get_books_by_genre(genre: str, top_k: int = 25) -> Tuple[List[str], List[float]]:
    """
    Genre-based recommendations using Pinecone: get seed books in this genre from Supabase,
    compute a genre centroid from their vectors, then query Pinecone for similar books.
    Returns (book_ids, similarity_scores).
    """
    try:
        # 1. Get seed book IDs in this genre from Supabase
        async with httpx.AsyncClient() as client:
            contains_val = 'cs.{"' + genre.replace('"', '\\"') + '"}'
            params = {"select": "id", "categories": contains_val, "limit": 50}
            resp = await client.get(f"{SUPABASE_URL}/rest/v1/books", headers=headers, params=params)
            resp.raise_for_status()
            books = resp.json()
        if not books:
            return [], []
        seed_ids = [str(b["id"]) for b in books if b.get("id")]
        if not seed_ids:
            return [], []

        # 2. Fetch seed book vectors from Pinecone and compute genre centroid
        fetch_result = pinecone_book_index.fetch(ids=seed_ids, namespace="__default__")
        vectors = getattr(fetch_result, "vectors", None) or {}
        vec_list = []
        for bid in seed_ids:
            rec = vectors.get(bid)
            if rec and getattr(rec, "values", None):
                vec_list.append(np.array(list(rec.values)))
        if not vec_list:
            # No vectors in Pinecone for this genre; fallback to seed IDs with uniform score
            return seed_ids[:top_k], [1.0] * min(len(seed_ids), top_k)
        centroid = np.mean(vec_list, axis=0)

        # 3. Query Pinecone for books similar to the genre centroid
        query_result = pinecone_book_index.query(
            vector=centroid.tolist(),
            top_k=top_k,
            namespace="__default__",
            include_metadata=False,
        )
        book_ids = []
        scores = []
        for m in (query_result.matches or []):
            bid = m.get("_id") or m.get("id") if hasattr(m, "get") else (getattr(m, "id", None) or getattr(m, "_id", None))
            score = m.get("score") if hasattr(m, "get") else getattr(m, "score", None)
            if bid is not None:
                book_ids.append(str(bid))
                scores.append(float(score) if score is not None else 1.0)
        return book_ids, scores
    except Exception as e:
        logger.warning(f"Pinecone genre similarity failed for {genre}: {e}")
        return [], []


def _get_currently_reading_book_ids_sync(user_id: str, limit: int = 20) -> List[str]:
    """
    Books the user has started reading (has at least one page_turn event).
    Returns list of book ids, most recently turned page first.
    """
    try:
        client = MongoClient(os.environ["MONGO_URI"])
        col = client["click_stream"]["events"]
        pipeline = [
            {"$match": {"user_id": user_id, "event_type": "page_turn"}},
            {"$sort": {"received_at": -1}},
            {"$group": {"_id": "$item_id", "last_at": {"$first": "$received_at"}}},
            {"$sort": {"last_at": -1}},
            {"$limit": limit},
            {"$project": {"_id": 1}}
        ]
        return [doc["_id"] for doc in col.aggregate(pipeline)]
    except Exception as e:
        logger.warning(f"MongoDB currently reading failed: {e}")
        return []


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
    """
    Returns recommendations grouped by algorithm/category in Netflix-style format.
    Always served from Redis cache (1-year TTL). On cache miss, computes once and stores.
    """
    user_id = str(current_user["id"])

    # Always try cache first (cache does not store Trending Now; we inject from global Redis)
    cached = get_cached_recommendations(user_id)
    if cached is not None:
        logger.info(f"Serving combined recommendations from cache for user {user_id}")
        trending = await _get_trending_now_category(top_k=50)
        categories_list = list(cached.get("categories") or [])
        categories_list = [c for c in categories_list if c.get("category") != "Trending Now"]
        if trending.get("books"):
            categories_list.append(trending)
        cached["categories"] = categories_list
        print(f"Serving combined recommendations from cache for user {user_id}: {cached}")
        return cached

    # Cache miss: compute full recommendations, then store and return
    loop = asyncio.get_running_loop()
    currently_reading_ids = await loop.run_in_executor(
        None, _get_currently_reading_book_ids_sync, user_id, 20
    )
    print(f"Cache miss: generating recommendations for user {user_id}")
    
    # Content-based
    cb_book_ids, cb_scores = await cbr.dense_vector_recommendation(user_id, top_k=50)
    
    # Collaborative filtering (ALS)
    als_book_ids, cf_scores = await cf.recommend_als_books(user_id, top_k=50)
    
    # Graph-based
    loop = asyncio.get_running_loop()
    graph_results = await loop.run_in_executor(None, graph_recommend_books, user_id, 50)
    graph_book_ids = [bid for bid, _ in graph_results]
    graph_scores = [score for _, score in graph_results]
    
    # Session-based (SASRec)
    session_book_ids, session_scores = recommend_for_session(user_id, top_k=50)
    
    # Popularity-based
    pop_book_ids, pop_scores = await get_popularity_recommendations(user_id, top_k=50)
    
    print(f"Got {len(cb_book_ids)} CB, {len(als_book_ids)} ALS, {len(graph_book_ids)} Graph, "
          f"{len(session_book_ids)} Session, {len(pop_book_ids)} Popularity books")

    # LinUCB: merge all candidates and rank with contextual bandit
    all_candidates = list(set(
        cb_book_ids + als_book_ids + graph_book_ids +
        session_book_ids + pop_book_ids
    ))
    print(f"All candidates: {all_candidates} with length {len(all_candidates)}")
    linucb_book_ids, linucb_scores = linucb_ranker.get_linucb_ranked(all_candidates, user_id)

    # Build combined_scores_map for feature engineering (Feast integration)
    combined_scores_map.clear()
    add_scores(cb_book_ids[:100], cb_scores[:100] if len(cb_scores) >= 100 else cb_scores, "content")
    add_scores(als_book_ids[:100], cf_scores[:100] if len(cf_scores) >= 100 else cf_scores, "als")
    add_scores(graph_book_ids[:100], graph_scores[:100] if len(graph_scores) >= 100 else graph_scores, "graph")
    add_scores(session_book_ids[:100], session_scores[:100] if len(session_scores) >= 100 else session_scores, "session")
    add_scores(pop_book_ids[:100], pop_scores[:100] if len(pop_scores) >= 100 else pop_scores, "popularity")
    add_scores(linucb_book_ids[:100], linucb_scores[:100] if len(linucb_scores) >= 100 else linucb_scores, "linucb")

    # Feast integration: engineer features and store to feature store (S3)
    feature_entities_from_batch = None
    candidate_ids = list(combined_scores_map.keys())
    try:
        session_context = await build_session_context(current_user, request)
        if candidate_ids:
            feature_entities_from_batch = await feature_service.engineer_features_batch(
                user_id, candidate_ids, combined_scores_map, session_context
            )
            logger.info(f"Feast: stored features for {len(candidate_ids)} user-book pairs")
            # Push offline store (S3) to online store (Redis) so get_online_features sees latest data
            try:
                await loop.run_in_executor(None, feature_service.materialize_offline_to_online)
            except Exception as mat_e:
                logger.warning(f"Feast materialize_incremental failed (non-fatal): {mat_e}")
    except Exception as e:
        logger.warning(f"Feast feature engineering failed (non-fatal): {e}")

    # Optional: retrieve features from Feast online store for LTR ranking
    feast_features_df = None
    try:
        feast_features_df = get_feast_online_features(user_id, candidate_ids or [])
    except Exception as e:
        logger.debug(f"Feast get_online_features skipped: {e}")

    # Use in-memory features for LTR when online store returns all None (e.g. materialize wrote 0 keys)
    if feast_features_df is not None and not feast_features_df.empty:
        feature_cols = [c for c in feast_features_df.columns if c != "user_book" and not c.startswith("event_timestamp")]
        if feature_cols and feast_features_df[feature_cols].isna().all().all():
            if feature_entities_from_batch:
                feast_features_df = pd.DataFrame(feature_entities_from_batch)
                logger.info("LTR using in-memory features (Feast online store had no values)")
            else:
                feast_features_df = None
                print("feature_entities_from_batch is None")

    # Learning-to-Rank: score candidates with XGBoost LambdaRank, add "Top Picks" category
    ltr_book_ids, ltr_scores = [], []
    if feast_features_df is not None and not feast_features_df.empty:
        try:
            ltr_book_ids, ltr_scores = ltr_rank_candidates(user_id, feast_features_df, top_k=100)
            if ltr_book_ids:
                logger.info(f"LTR ranked {len(ltr_book_ids)} books for user {user_id}")
        except Exception as e:
            logger.warning(f"LTR ranking failed (non-fatal): {e}")

    # 2. Fetch book metadata for each category (order: Continue Reading, then per-genre, then algorithms + Top Picks)
    categories = {}
    if currently_reading_ids:
        categories["Continue Reading"] = {
            "book_ids": currently_reading_ids,
            "scores": [1.0] * len(currently_reading_ids),
            "description": "Pick up where you left off"
        }
    categories["Content-Based Recommendations"] = {
        "book_ids": cb_book_ids,
        "scores": cb_scores,
        "description": "Books similar to your preferences"
    }
    categories["Collaborative Filtering"] = {
        "book_ids": als_book_ids,
        "scores": cf_scores,
        "description": "Readers with similar tastes also enjoyed"
    }
    categories["Social Recommendations"] = {
        "book_ids": graph_book_ids,
        "scores": graph_scores,
        "description": "Based on your social network"
    }
    categories["Session-Based"] = {
        "book_ids": session_book_ids,
        "scores": session_scores,
        "description": "Based on your recent activity"
    }
    categories["Trending Now"] = {
        "book_ids": pop_book_ids,
        "scores": pop_scores,
        "description": "Popular books right now"
    }
    categories["For You (LinUCB)"] = {
        "book_ids": linucb_book_ids,
        "scores": linucb_scores,
        "description": "Personalized picks with exploration"
    }
    # Top Picks from LTR (always set category; empty when LTR has no results)
    categories["Top Picks"] = {
        "book_ids": ltr_book_ids,
        "scores": ltr_scores,
        "description": "Recommended for you by our ranking model"
    }

    # Fetch all unique books (include LinUCB, continue reading, genre rows, Top Picks)
    all_book_ids = list(set(
        cb_book_ids + als_book_ids + graph_book_ids +
        session_book_ids + pop_book_ids + linucb_book_ids +
        currently_reading_ids + ltr_book_ids
    ))
    books_dict = {book["id"]: book for book in await get_books_by_ids(all_book_ids)}
    
    # Build category-based recommendations
    category_recommendations = []
    for category_name, category_data in categories.items():
        if not category_data["book_ids"]:
            continue
            
        books = []
        for book_id, score in zip(category_data["book_ids"], category_data["scores"]):
            if book_id in books_dict:
                book = books_dict[book_id].copy()
                book["score"] = float(score)
                books.append(book)
        
        if books:
            category_recommendations.append({
                "category": category_name,
                "description": category_data["description"],
                "books": books
            })
    
    payload = {"categories": category_recommendations}
    # Don't store Trending Now per user; it always comes from global Redis (popularity:trending:*)
    to_cache = {"categories": [c for c in category_recommendations if c.get("category") != "Trending Now"]}
    set_cached_recommendations(user_id, to_cache)
    print(f"Returning {len(category_recommendations)} categories with recommendations")
    return payload


def add_scores(book_ids, scores, source_name):
    for bid, score in zip(book_ids, scores):
        combined_scores_map[bid][f"{source_name}_score"] = float(score)


def _build_category_payload(
    category_name: str,
    description: str,
    book_ids: List[str],
    scores: List[float],
    books_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one category object for cache: {category, description, books} with full book objects + score."""
    books = []
    for bid, score in zip(book_ids, scores):
        if bid in books_dict:
            b = books_dict[bid].copy()
            b["score"] = float(score)
            books.append(b)
    return {"category": category_name, "description": description, "books": books}


# ---------- Internal refresh endpoints (called by background pods) ----------


@app.post("/internal/recommend/refresh-content-based")
async def internal_refresh_content_based(
    body: RefreshBody,
    _: bool = Depends(verify_internal_key),
):
    """
    Run content-based recommendation and update cache. Call when user preferences (genre/author) change.
    """
    user_id = body.user_id
    try:
        cb_book_ids, cb_scores = await cbr.dense_vector_recommendation(user_id, top_k=50)
        if not cb_book_ids:
            update_cached_category(
                user_id,
                "Content-Based Recommendations",
                _build_category_payload(
                    "Content-Based Recommendations",
                    "Books similar to your preferences",
                    [], [], {},
                ),
            )
            return {"status": "ok", "category": "Content-Based Recommendations", "count": 0}
        # Write features to S3 for LTR training (content_score only for this refresh)
        try:
            scores_map = defaultdict(dict)
            for bid, score in zip(cb_book_ids, cb_scores):
                scores_map[bid]["content_score"] = float(score)
                scores_map[bid].setdefault("als_score", 0.0)
                scores_map[bid].setdefault("graph_score", 0.0)
                scores_map[bid].setdefault("session_score", 0.0)
                scores_map[bid].setdefault("popularity_score", 0.0)
                scores_map[bid].setdefault("linucb_score", 0.5)
            _ = await feature_service.engineer_features_batch(
                user_id, cb_book_ids, dict(scores_map), _minimal_session_context()
            )
            logger.info(f"Refresh content-based: wrote features to S3 for {len(cb_book_ids)} books")
            try:
                await asyncio.get_running_loop().run_in_executor(None, feature_service.materialize_offline_to_online)
            except Exception as mat_e:
                logger.warning(f"Feast materialize_incremental failed (non-fatal): {mat_e}")
        except Exception as e:
            logger.warning(f"Refresh content-based: feature write to S3 failed (non-fatal): {e}")
        books_list = await get_books_by_ids(cb_book_ids)
        books_dict = {str(b.get("id") or b.get("_id")): b for b in books_list}
        payload = _build_category_payload(
            "Content-Based Recommendations",
            "Books similar to your preferences",
            cb_book_ids, cb_scores, books_dict,
        )
        update_cached_category(user_id, "Content-Based Recommendations", payload)
        return {"status": "ok", "category": "Content-Based Recommendations", "count": len(payload["books"])}
    except Exception as e:
        logger.exception(f"Refresh content-based failed for {user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internal/recommend/refresh-on-logout")
async def internal_refresh_on_logout(
    body: RefreshBody,
    _: bool = Depends(verify_internal_key),
):
    """
    Run collaborative_filtering, SASRec, LinUCB, and currently-reading; update cache.
    Call when user logs out (so next login gets fresh CF/session/LinUCB/continue-reading).
    """
    user_id = body.user_id
    loop = asyncio.get_running_loop()
    try:
        # Collaborative Filtering
        als_book_ids, cf_scores = await cf.recommend_als_books(user_id, top_k=50)
        all_book_ids = list(als_book_ids)
        # Session-based (SASRec)
        session_book_ids, session_scores = recommend_for_session(user_id, top_k=50)
        all_book_ids.extend(session_book_ids)
        # Currently reading
        currently_reading_ids = await loop.run_in_executor(
            None, _get_currently_reading_book_ids_sync, user_id, 20
        )
        all_book_ids.extend(currently_reading_ids)
        # LinUCB needs candidates: use ALS + session + currently_reading
        linucb_candidates = list(set(als_book_ids + session_book_ids + currently_reading_ids))
        linucb_book_ids, linucb_scores = linucb_ranker.get_linucb_ranked(linucb_candidates, user_id)
        all_book_ids.extend(linucb_book_ids)
        candidate_ids = list(set(all_book_ids))

        # Write features to S3 for LTR training (from ALS, session, LinUCB, continue-reading)
        feature_entities_from_batch = None
        try:
            scores_map = defaultdict(dict)
            for bid, score in zip(als_book_ids, cf_scores):
                scores_map[bid]["als_score"] = float(score)
            for bid, score in zip(session_book_ids, session_scores):
                scores_map[bid]["session_score"] = float(score)
            for bid, score in zip(linucb_book_ids, linucb_scores):
                scores_map[bid]["linucb_score"] = float(score)
            for bid in currently_reading_ids:
                scores_map[bid]["session_score"] = 1.0
            for bid in candidate_ids:
                scores_map[bid].setdefault("content_score", 0.0)
                scores_map[bid].setdefault("als_score", 0.0)
                scores_map[bid].setdefault("graph_score", 0.0)
                scores_map[bid].setdefault("session_score", 0.0)
                scores_map[bid].setdefault("popularity_score", 0.0)
                scores_map[bid].setdefault("linucb_score", 0.5)
            feature_entities_from_batch = await feature_service.engineer_features_batch(
                user_id, candidate_ids, dict(scores_map), _minimal_session_context()
            )
            logger.info(f"Refresh on-logout: wrote features to S3 for {len(candidate_ids)} books")
            try:
                await asyncio.get_running_loop().run_in_executor(None, feature_service.materialize_offline_to_online)
            except Exception as mat_e:
                logger.warning(f"Feast materialize_incremental failed (non-fatal): {mat_e}")
        except Exception as e:
            logger.warning(f"Refresh on-logout: feature write to S3 failed (non-fatal): {e}")

        books_list = await get_books_by_ids(candidate_ids[:200])
        books_dict = {str(b.get("id") or b.get("_id")): b for b in books_list}

        # Learning-to-Rank: get features (Feast online or from engineered batch), rank, update "Top Picks" in cache
        ltr_ids, ltr_scores = [], []
        try:
            feast_df = get_feast_online_features(user_id, candidate_ids[:200])
            if feast_df is not None and not feast_df.empty:
                ltr_ids, ltr_scores = ltr_rank_candidates(user_id, feast_df, top_k=100)
            elif feature_entities_from_batch:
                # Fallback: use freshly engineered features as DataFrame for LTR
                ltr_feature_df = pd.DataFrame(feature_entities_from_batch)
                if not ltr_feature_df.empty:
                    ltr_ids, ltr_scores = ltr_rank_candidates(user_id, ltr_feature_df, top_k=100)
            if ltr_ids:
                update_cached_category(
                    user_id, "Top Picks",
                    _build_category_payload(
                        "Top Picks",
                        "Recommended for you by our ranking model",
                        ltr_ids, ltr_scores, books_dict,
                    ),
                )
        except Exception as e:
            logger.warning(f"LTR refresh on logout failed for {user_id}: {e}")

        update_cached_category(
            user_id, "Collaborative Filtering",
            _build_category_payload(
                "Collaborative Filtering",
                "Readers with similar tastes also enjoyed",
                als_book_ids, cf_scores, books_dict,
            ),
        )
        update_cached_category(
            user_id, "Session-Based",
            _build_category_payload(
                "Session-Based",
                "Based on your recent activity",
                session_book_ids, session_scores, books_dict,
            ),
        )
        update_cached_category(
            user_id, "For You (LinUCB)",
            _build_category_payload(
                "For You (LinUCB)",
                "Personalized picks with exploration",
                linucb_book_ids, linucb_scores, books_dict,
            ),
        )
        update_cached_category(
            user_id, "Continue Reading",
            _build_category_payload(
                "Continue Reading",
                "Pick up where you left off",
                currently_reading_ids,
                [1.0] * len(currently_reading_ids),
                books_dict,
            ),
        )
        updated = ["Collaborative Filtering", "Session-Based", "For You (LinUCB)", "Continue Reading", "Top Picks"]
        return {"status": "ok", "updated": updated}
    except Exception as e:
        logger.exception(f"Refresh on-logout failed for {user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internal/recommend/refresh-graph")
async def internal_refresh_graph(
    body: RefreshBody,
    _: bool = Depends(verify_internal_key),
):
    """
    Run graph recommendation and update cache. Call when user follows or unfollows another user.
    """
    user_id = body.user_id
    loop = asyncio.get_running_loop()
    try:
        graph_results = await loop.run_in_executor(None, graph_recommend_books, user_id, 50)
        graph_book_ids = [bid for bid, _ in graph_results]
        graph_scores = [score for _, score in graph_results]
        if not graph_book_ids:
            update_cached_category(
                user_id, "Social Recommendations",
                _build_category_payload("Social Recommendations", "Based on your social network", [], [], {}),
            )
            return {"status": "ok", "category": "Social Recommendations", "count": 0}
        # Write features to S3 for LTR training (graph_score only for this refresh)
        try:
            scores_map = defaultdict(dict)
            for bid, score in zip(graph_book_ids, graph_scores):
                scores_map[bid]["graph_score"] = float(score)
                scores_map[bid].setdefault("content_score", 0.0)
                scores_map[bid].setdefault("als_score", 0.0)
                scores_map[bid].setdefault("session_score", 0.0)
                scores_map[bid].setdefault("popularity_score", 0.0)
                scores_map[bid].setdefault("linucb_score", 0.5)
            _ = await feature_service.engineer_features_batch(
                user_id, graph_book_ids, dict(scores_map), _minimal_session_context()
            )
            logger.info(f"Refresh graph: wrote features to S3 for {len(graph_book_ids)} books")
            try:
                await asyncio.get_running_loop().run_in_executor(None, feature_service.materialize_offline_to_online)
            except Exception as mat_e:
                logger.warning(f"Feast materialize_incremental failed (non-fatal): {mat_e}")
        except Exception as e:
            logger.warning(f"Refresh graph: feature write to S3 failed (non-fatal): {e}")
        books_list = await get_books_by_ids(graph_book_ids)
        books_dict = {str(b.get("id") or b.get("_id")): b for b in books_list}
        payload = _build_category_payload(
            "Social Recommendations",
            "Based on your social network",
            graph_book_ids, graph_scores, books_dict,
        )
        update_cached_category(user_id, "Social Recommendations", payload)
        return {"status": "ok", "category": "Social Recommendations", "count": len(payload["books"])}
    except Exception as e:
        logger.exception(f"Refresh graph failed for {user_id}")
        raise HTTPException(status_code=500, detail=str(e))

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


def _minimal_session_context() -> Dict[str, Any]:
    """Minimal session context for internal refresh endpoints (no Request). Used when writing features to S3."""
    return {
        "device": "desktop",
        "hour_of_day": datetime.now().hour,
        "user_agent": "",
    }


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


def get_feast_online_features(user_id: str, book_ids: List[str]):
    """
    Retrieve features from Feast online store for (user_id, book_id) pairs.
    Uses get_feast_repo_path() so FEAST_REDIS_URL is applied (feast-redis in K8s).
    Returns a DataFrame or None if Feast is not available or store is not configured.
    """
    if not book_ids:
        return None
    from feature_engineering.feature_service import get_feast_repo_path
    repo_path = get_feast_repo_path()
    if not os.path.isdir(repo_path):
        logger.debug(f"Feast repo not found at {repo_path}")
        return None
    store = FeatureStore(repo_path=repo_path)
    entity_keys = [f"{user_id}_{bid}" for bid in book_ids]
    entity_df = pd.DataFrame({"user_book": entity_keys})
    feature_vector = store.get_online_features(
        entity_rows=entity_df.to_dict("records"),
        features=[
            "user_book_features:query_id",
            "user_book_features:retrieval_0",
            "user_book_features:retrieval_1",
            "user_book_features:retrieval_2",
            "user_book_features:retrieval_3",
            "user_book_features:retrieval_4",
            "user_book_features:retrieval_5",
            "user_book_features:genre_match_count",
            "user_book_features:author_match",
            "user_book_features:language_match",
            "user_book_features:avg_rating",
            "user_book_features:user_rating",
            "user_book_features:avg_rating_diff",
            "user_book_features:rating_count",
            "user_book_features:user_pref_strength",
            "user_book_features:friend_reads_count",
            "user_book_features:friend_avg_rating",
            "user_book_features:author_following",
            "user_book_features:mutual_likes",
            "user_book_features:social_proximity",
            "user_book_features:session_position",
            "user_book_features:session_genre_drift",
            "user_book_features:time_since_last_action",
            "user_book_features:is_mobile",
            "user_book_features:is_desktop",
            "user_book_features:is_tablet",
            "user_book_features:session_length",
            "user_book_features:global_pop_rank",
            "user_book_features:trending_score",
            "user_book_features:intra_list_diversity",
        ],
    ).to_df()
    return feature_vector
