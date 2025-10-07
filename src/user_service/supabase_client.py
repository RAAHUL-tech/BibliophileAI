import os
import httpx
from typing import Optional
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