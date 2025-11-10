from http.client import HTTPException
import os
import httpx
from typing import Optional, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL =  os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

async def get_user_by_username(username: str) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        params = {"username": f"eq.{username}"}
        url = f"{SUPABASE_URL}/rest/v1/users"
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None

async def create_user(user: dict) -> str:
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/users"
        response = await client.post(url, json=user, headers=headers)
        if response.is_error:
            print(f"Supabase user create failed: {response.status_code} {response.text}")
        response.raise_for_status()
        return user["username"]

async def get_user_by_email(email: str) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        params = {"email": f"eq.{email}"}
        url = f"{SUPABASE_URL}/rest/v1/users"
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None

async def get_user_profile_by_id(user_id: str) -> dict:
    """
    Fetch the user profile from Supabase users table by user_id.
    """
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/users"
        params = {"id": f"eq.{user_id}"}
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}

async def get_preferences_by_user_id(user_id: int) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/user_preferences"
        params = {"user_id": f"eq.{user_id}"}
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None

async def create_preferences(user_id: str, genres: list[str], authors: list[str]) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/user_preferences"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"user_id": user_id, "genres": genres, "authors": authors},
            headers=headers
        )
        response.raise_for_status()
        # Handle possibility of empty response (Created, but no content)
        if response.status_code == 201 and not response.content:
            return {"user_id": user_id, "genres": genres, "authors": authors}
        data = response.json()
        return data[0] if isinstance(data, list) else data

async def update_preferences(
    user_id: str,
    genres: list[str],
    authors: list[str]
) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/user_preferences"
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            url,
            json={
                "genres": genres,
                "authors": authors,
                "updated_at": updated_at
            },
            headers=headers,
            params={"user_id": f"eq.{user_id}"}
        )
        response.raise_for_status()
        # PATCH may return 204 No Content
        logger.info(f"PATCH {user_id}: {response.status_code} {response.text}")
        if response.status_code == 204 or not response.content:
            return {"user_id": user_id, "genres": genres, "authors": authors, "updated_at": updated_at}
        data = response.json()
        return data[0] if isinstance(data, list) else data


async def update_user_profile(user_id: str, age: int, pincode: str) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/users"
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            url,
            json={"age": age, "pincode": pincode, "updated_at": updated_at},
            headers=headers,
            params={"id": f"eq.{user_id}"}
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {"user_id": user_id, "age": age, "pincode": pincode}
        data = response.json()
        return data[0] if isinstance(data, list) else data


async def get_popular_authors_from_db() -> list[str]:
    """
    Get the most popular authors from Supabase using RPC.
    Returns a list of author names.
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/rpc/get_popular_authors"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json={}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract just the author names from the results
            authors = [item["author"] for item in data]
            return authors
            
    except Exception as e:
        logger.error(f"Error fetching popular authors from Supabase: {e}")
        return []
    

async def create_session(
    user_id: str,
    device: str,
    ip_address: str,
    user_agent: str,
    metadata: Optional[Dict] = None
) -> str:
    url = f"{SUPABASE_URL}/rest/v1/sessions"
    payload = {
        "user_id": user_id,
        "device": device,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "metadata": metadata or {},
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "is_active": True
    }
    # Add Prefer header to get the inserted row back
    headers_with_prefer = {
        **headers,
        "Prefer": "return=representation"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers_with_prefer)

    # Always check status
    if response.status_code != 201:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {response.status_code} {response.text}")

    # Handle empty response
    if not response.content:
        # If no body, return a placeholder or generate a client-side ID
        # In production, you might want to fetch the session later by user_id
        return "fallback_session_id_" + user_id  # or use uuid

    try:
        data = response.json()
        return data[0]["session_id"] if isinstance(data, list) else data["session_id"]
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=500, detail="Invalid session response from database")


async def end_session(session_id: str):
    url = f"{SUPABASE_URL}/rest/v1/sessions"
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            url,
            json={"is_active": False, "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},
            headers=headers,
            params={"session_id": f"eq.{session_id}"}
        )
        response.raise_for_status()

async def get_reviews_and_avg_rating_from_db(book_id: str) -> dict:
    """
    Fetch average rating and list of reviews for a book.
    Returns:
        {
            "avg_rating": 4.2,
            "reviews": [
                {"user": "alice", "rating": 5, "text": "Great book!"},
                ...
            ]
        }
    """
    async with httpx.AsyncClient() as client:
        # Query to get average rating and reviews with username
        query = """
            SELECT
                r.content,
                r.reviewed_at,
                rt.rating,
                u.username
            FROM review r
            JOIN rating rt ON r.book_id = rt.book_id AND r.user_id = rt.user_id
            JOIN users u ON r.user_id = u.id
            WHERE r.book_id = :book_id
            ORDER BY r.reviewed_at DESC
        """

        url = f"{SUPABASE_URL}/rest/v1/rpc/get_book_reviews_and_ratings"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        response = await client.post(
            url,
            json={
                "book_id": book_id
            },
            headers=headers
        )

        if response.status_code != 200:
            print("Supabase RPC error:", response.status_code, response.text)
            return {"avg_rating": 0, "reviews": []}

        rows = response.json()

        if not rows:
            return {"avg_rating": 0, "reviews": []}

        # Extract ratings for average
        ratings = [row["rating"] for row in rows]
        avg_rating = round(sum(ratings) / len(ratings), 2)

        # Format reviews
        reviews = [
            {
                "user": row["username"],
                "rating": row["rating"],
                "text": row["content"]
            }
            for row in rows
        ]

        return {
            "avg_rating": avg_rating,
            "reviews": reviews
        }

async def save_review_and_rating_to_db(
    user_id: str,
    book_id: str,
    rating: int,
    content: str
) -> bool:
    """
    Save or update a user's review and rating for a book.
    Uses Supabase REST API with upsert on conflict.
    """
    async with httpx.AsyncClient() as client:
        # Insert or update review
        review_res = await client.post(
            f"{SUPABASE_URL}/rest/v1/review",
            json={
                "user_id": user_id,
                "book_id": book_id,
                "content": content,
                "updated_at": datetime.now().isoformat()
            },
            headers=headers,
            params={"user_id": f"eq.{user_id}", "book_id": f"eq.{book_id}"},
            # This will upsert if row exists (requires unique constraint on user_id+book_id)
        )

        if review_res.status_code not in (200, 201):
            print("Review error:", review_res.status_code, review_res.text)
            return False

        # Insert or update rating
        rating_res = await client.post(
            f"{SUPABASE_URL}/rest/v1/rating",
            json={
                "user_id": user_id,
                "book_id": book_id,
                "rating": rating
            },
            headers=headers,
            params={"user_id": f"eq.{user_id}", "book_id": f"eq.{book_id}"}
        )

        if rating_res.status_code not in (200, 201):
            print("Rating error:", rating_res.status_code, rating_res.text)
            return False

        return True

async def is_bookmarked_by_user(user_id: str, book_id: str) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/user_bookmarks"
    params = {
        "user_id": f"eq.{user_id}",
        "book_id": f"eq.{book_id}",
        "select": "user_id"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(url, headers=headers, params=params)
        if res.status_code != 200:
            print("Supabase bookmark check error:", res.status_code, res.text)
            return False
        data = res.json()
        return bool(data and len(data) > 0)

async def add_user_bookmark(user_id: str, book_id: str):
    url = f"{SUPABASE_URL}/rest/v1/user_bookmarks"
    payload = {"user_id": user_id, "book_id": book_id, "bookmarked_at": datetime.now().isoformat()}
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code not in (200, 201):
            print("Supabase add bookmark error:", res.status_code, res.text)
            raise Exception("Failed to add bookmark")
        return {"user_id": user_id, "book_id": book_id}

async def remove_user_bookmark(user_id: str, book_id: str):
    url = f"{SUPABASE_URL}/rest/v1/user_bookmarks"
    params = {
        "user_id": f"eq.{user_id}",
        "book_id": f"eq.{book_id}"
    }
    async with httpx.AsyncClient() as client:
        res = await client.delete(url, headers=headers, params=params)
        if res.status_code not in (200, 204):
            print("Supabase remove bookmark error:", res.status_code, res.text)
            raise Exception("Failed to remove bookmark")
        return {"user_id": user_id, "book_id": book_id}

async def get_books_by_ids(ids: list[str]) -> list[dict]:
    """
    Fetch book details from Supabase books table by list of IDs.
    """
    if not ids:
        return []
    async with httpx.AsyncClient() as client:
        # Format: id=in.("id1","id2",...)
        idlist = ",".join([f'"{bid}"' for bid in ids])
        fields = "id,title,authors,categories,thumbnail_url,download_link"
        url = f"{SUPABASE_URL}/rest/v1/books"
        params = {
            "select": fields,
            "id": f"in.({idlist})"
        }
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

async def get_user_bookmark_ids(user_id: str) -> list[str]:
    """
    Fetch all book_ids bookmarked by the user.
    """
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/user_bookmarks"
        params = {
            "user_id": f"eq.{user_id}",
            "select": "book_id",
            "order": "bookmarked_at.desc"
        }
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return [item["book_id"] for item in resp.json()]
        elif resp.status_code == 404:
            return []
        else:
            resp.raise_for_status()
    return []


async def get_current_active_session_id(user_id: str) -> str:
    """
    Returns the session_id of the latest active session for a user.
    If multiple are active, picks the most recent last_activity.
    Returns None if there are no active sessions.
    """
    url = f"{SUPABASE_URL}/rest/v1/sessions"
    params = {
        "user_id": f"eq.{user_id}",
        "is_active": "eq.true",          # Filters only active sessions
        "order": "last_activity.desc",   # Most recent first
        "limit": 1,                      # Only one needed
        "select": "session_id,last_activity"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print("Supabase session query error:", resp.status_code, resp.text)
            return None
        data = resp.json()
        if data and len(data) > 0:
            return data[0]
        return None

