import os
import httpx
from typing import Optional


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

async def create_or_update_preferences(user_id: int, genres: list[str]) -> dict:
    genres_str = ",".join(genres)
    url = f"{SUPABASE_URL}/rest/v1/user_preferences"
    async with httpx.AsyncClient() as client:
        check = await client.get(url, headers=headers, params={"user_id": f"eq.{user_id}"})
        check.raise_for_status()
        existing = check.json()
        if existing:
            pref_id = existing[0]["id"]
            response = await client.patch(
                url,
                json={"genres": genres_str},
                headers=headers,
                params={"id": f"eq.{pref_id}"},
            )
        else:
            response = await client.post(
                url,
                json={"user_id": user_id, "genres": genres_str},
                headers=headers
            )
        response.raise_for_status()
        if response.content:
            data = response.json()
        else:
            data = None
        return data[0] if isinstance(data, list) else data


async def get_preferences_by_user_id(user_id: int) -> Optional[dict]:
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/user_preferences"
        params = {"user_id": f"eq.{user_id}"}
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
