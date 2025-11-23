import os
import httpx
from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import jwt
import content_based_recommendation as cbr
import random
import collaborative_filtering as cf
from typing import List

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


import random

@app.get("/api/v1/recommend/combined")
async def recommend_combined(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["id"])

    # 1. Get IDs and relevance scores
    cb_book_ids, cb_scores = await cbr.dense_vector_recommendation(user_id, top_k=50)
    als_book_ids, cf_scores = await cf.recommend_als_books(user_id, top_k=50)

    # 2. Deduplicate: remove any ALS IDs that are also in CB
    cb_set = set(cb_book_ids)
    als_unique = [bid for bid in als_book_ids if bid not in cb_set]
    # Indices for ALS unique results
    als_unique_indices = [i for i, bid in enumerate(als_book_ids) if bid not in cb_set]

    # 3. Take top 25 from each, preserving matching scores
    cb_final = cb_book_ids[:25]
    cb_final_scores = cb_scores[:25]
    als_final = als_unique[:25]
    # Get corresponding scores for deduped ALS choices
    als_final_scores = [cf_scores[i] for i in als_unique_indices[:25]]

    # 4. Combine and shuffle
    all_ids = cb_final + als_final
    all_scores = cb_final_scores + als_final_scores
    # Pair IDs and scores together for shuffle
    combined = list(zip(all_ids, all_scores))
    random.shuffle(combined)
    combined = combined[:50]
    shuffled_ids, shuffled_scores = zip(*combined) if combined else ([], [])

    # 5. Fetch metadata for all recommended books
    books = await get_books_by_ids(list(shuffled_ids))

    # Merge scores with book metadata
    book_score_dict = dict(zip(shuffled_ids, shuffled_scores))
    recommendations = []
    for book in books:
        # Try both "id" and "_id" fields for robustness
        book_id = book.get("id") or book.get("_id")
        if book_id in book_score_dict:
            book["relevance_score"] = book_score_dict[book_id]
        recommendations.append(book)

    return {"recommendations": recommendations}
