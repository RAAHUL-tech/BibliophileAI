from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
import os
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from schemas import UserCreate, GoogleToken, UserPreferences
from supabase_client import (
    get_user_by_username,
    get_user_by_email,
    create_user,
    update_preferences,
    create_preferences,
    get_preferences_by_user_id,
    update_user_profile, get_popular_authors_from_db
)
from auth import create_access_token, decode_access_token
from typing import Dict
import hashlib
from user_embeddings import create_user_embedding_vectors
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def prehash_password(password: str) -> str:
    # Pre-hash full password using SHA-256 then hex encode to a string
    sha256_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return sha256_hash

@app.post("/register")
async def register(user: UserCreate):
    existing_user = await get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    safe_password = prehash_password(user.password)
    hashed_password = pwd_context.hash(safe_password)
    user_data = {"username": user.username, "email": user.email, "hashed_password": hashed_password}
    new_user = await create_user(user_data)
    access_token = create_access_token(data={"sub": new_user})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_username(form_data.username)
    if not user or not user.get("hashed_password"):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    safe_password = prehash_password(form_data.password)
    if not pwd_context.verify(safe_password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/google-register")
async def google_register(token: GoogleToken):
    try:
        idinfo = id_token.verify_oauth2_token(token.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        name = idinfo.get("name", email.split("@")[0])
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")
        db_user = await get_user_by_email(email)
        if db_user:
            raise HTTPException(status_code=400, detail="User already registered")
        user_data = {"username": name, "email": email, "hashed_password": None}
        user = await create_user(user_data)
        access_token = create_access_token(data={"sub": user})
        return {"access_token": access_token, "token_type": "bearer"}
    except ValueError as ex:
        print("Google token error:", ex)
        raise HTTPException(status_code=400, detail="Invalid Google token")

@app.post("/google-login")
async def google_login(token: GoogleToken):  
    try:
        idinfo = id_token.verify_oauth2_token(token.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")
        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=400, detail="User not registered")
        access_token = create_access_token(data={"sub": user["username"]})
        return {"access_token": access_token, "token_type": "bearer"}
    except ValueError as ex:
        print("Google token error:", ex)
        raise HTTPException(status_code=400, detail="Invalid Google token")

@app.post("/user/preferences")
async def save_preferences(
    prefs_in: UserPreferences, 
    current_user=Depends(get_current_user)
) -> Dict:
    """
    Save complete user preferences including genres, authors, pincode, and age.
    Updates both user_preferences and users tables in Supabase.
    """
    user_id = current_user["id"]
    
    try:
        # Update user_preferences table with genres and authors
        preferences_result = await create_preferences(
            user_id=user_id,
            genres=prefs_in.genres,
            authors=prefs_in.authors
        )
        
        if not preferences_result:
            logger.error(f"Failed to save preferences for user {user_id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save preferences"
            )
            
        # Update users table with age and pincode (as part of profile)
        await update_user_profile(
            user_id=user_id,
            age=prefs_in.age,
            pincode=prefs_in.pincode
        )
        logger.info(f"Prefenece result  for user {user_id}: {preferences_result}")
        # Create embedding vectors for recommendation system
        try:
            await create_user_embedding_vectors(user_id, prefs_in.genres, prefs_in.authors, prefs_in.age, prefs_in.pincode)
        except Exception as e:
            logger.warning(f"Error creating embedding for user {user_id}: {e}")
            # Don't fail the entire request if embedding creation fails
        
        return {
            "user_id": user_id,
            "genres": prefs_in.genres,
            "authors": prefs_in.authors,
            "age": prefs_in.age,
            "pincode": prefs_in.pincode,
            "updated_at": preferences_result.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid preferences data"
        )
    

@app.get("/user/preferences")
async def get_preferences(current_user=Depends(get_current_user)):
    """
    Get user preferences including both genres and authors.
    Returns empty arrays if no preferences exist.
    """
    prefs = await get_preferences_by_user_id(current_user["id"])
    if not prefs:
        return {
            "id": None,
            "user_id": current_user["id"], 
            "genres": [],
            "authors": []
        }
    
    # Extract arrays directly - Supabase returns proper array types
    genres_list = prefs.get("genres", []) or []
    authors_list = prefs.get("authors", []) or []
    
    return {
        "id": prefs["id"],
        "user_id": current_user["id"],
        "genres": genres_list,
        "authors": authors_list
    }

@app.patch("/user/preferences")
async def patch_preferences(
    prefs_in: UserPreferences,
    current_user=Depends(get_current_user)
) -> Dict:
    """
    Patch (update) user preferences for genres and authors only.
    """
    user_id = current_user["id"]

    # Make sure at least one field is provided:
    if prefs_in.genres is None and prefs_in.authors is None:
        raise HTTPException(status_code=400, detail="At least one preference field (genres or authors) required")

    # Get current preferences
    current = await get_preferences_by_user_id(user_id)
    if not current:
        raise HTTPException(status_code=404, detail="Preferences not found")

    # If field is not provided, keep old value
    genres = prefs_in.genres if prefs_in.genres is not None else current.get("genres", [])
    authors = prefs_in.authors if prefs_in.authors is not None else current.get("authors", [])

    # Update preferences only
    preferences_result = await update_preferences(
        user_id=user_id,
        genres=genres,
        authors=authors
    )

    if not preferences_result:
        logger.error(f"Failed to patch preferences for user {user_id}")
        raise HTTPException(status_code=500, detail="Failed to patch preferences")

    logger.info(f"PATCH preference result for user {user_id}: {preferences_result}")
    try:
        await create_user_embedding_vectors(user_id, prefs_in.genres, prefs_in.authors, prefs_in.age, prefs_in.pincode)
    except Exception as e:
        logger.warning(f"Error creating embedding for user {user_id}: {e}")
        
    return {
        "user_id": user_id,
        "genres": genres,
        "authors": authors,
        "updated_at": preferences_result.get("updated_at")
    }

@app.get("/user/profile")
async def get_user_profile(current_user=Depends(get_current_user)):
    """
    Get complete user profile information including username, email, age, and pincode.
    """
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "age": current_user.get("age"),
        "pincode": current_user.get("pincode")
    }

@app.get("/popular-authors")
async def get_popular_authors(current_user=Depends(get_current_user)):
    """
    Get the most popular authors based on book count.
    Returns top 20 authors ordered by popularity.
    """
    try:
        authors = await get_popular_authors_from_db()
        return {"authors": authors}
            
    except Exception as e:
        logger.error(f"Error in get_popular_authors route: {e}")
        return {"authors": []}