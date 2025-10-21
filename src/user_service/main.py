from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response, Query
import requests
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
import os
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from schemas import UserCreate, GoogleToken, UserPreferences, LogoutRequest, ProfilePreferences, SubmitReviewRequest, PageTurnEvent
from supabase_client import (
    get_user_by_username,
    get_user_by_email,
    create_user,
    is_bookmarked_by_user,
    add_user_bookmark,
    remove_user_bookmark,
    update_preferences, get_current_active_session_id,
    create_preferences, get_user_bookmark_ids, get_books_by_ids,
    get_preferences_by_user_id, save_review_and_rating_to_db,
    update_user_profile, get_popular_authors_from_db, end_session, create_session, get_reviews_and_avg_rating_from_db
)
from neo4j_client import (create_user_follows_users, create_user_read_book, create_user_bookmarked_book, update_user_profile_fields,
                          delete_user_bookmarked_book, create_user_rated_book, create_user_preferences, patch_user_preferences)
import httpx
from user_agents import parse
import uuid
from datetime import datetime
from auth import create_access_token, decode_access_token
from typing import Dict
import hashlib
from user_embeddings import create_user_embedding_vectors
import logging
from io import BytesIO
import zipfile
import mimetypes
from produce import send_click_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
S3_BUCKET_URL_TEMPLATE = "https://bibliophileai.s3.us-east-2.amazonaws.com/books-epub/{book_id}.epub"


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

@app.post("/api/v1/user/register")
async def register(request: Request, user: UserCreate):
    existing_user = await get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    safe_password = prehash_password(user.password)
    hashed_password = pwd_context.hash(safe_password)
    user_data = {"username": user.username, "email": user.email, "hashed_password": hashed_password}
    new_user = await create_user(user_data)
    access_token = create_access_token(data={"sub": new_user})
    user_agent = request.headers.get("user-agent", "")
    ip_address = request.client.host if request.client else ""
    ua = parse(user_agent)
    device = "mobile" if ua.is_mobile else "tablet" if ua.is_tablet else "desktop"
    session_id = await create_session(
        user_id=user["id"],
        device=device,
        ip_address=ip_address,
        user_agent=user_agent,
        metadata=None
    )
    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user["id"],
        "item_id": None,
        "event_type": "register",
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "duration": None,
        "metadata": {}
    }
    group_id = f"user_{user['id']}"
    send_click_event(event, group_id)
    return {"access_token": access_token, "token_type": "bearer", "session_id": session_id}

@app.post("/api/v1/user/login")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_username(form_data.username)
    if not user or not user.get("hashed_password"):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    safe_password = prehash_password(form_data.password)
    if not pwd_context.verify(safe_password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    user_agent = request.headers.get("user-agent", "")
    ip_address = request.client.host if request.client else ""
    ua = parse(user_agent)
    device = "mobile" if ua.is_mobile else "tablet" if ua.is_tablet else "desktop"
    session_id = await create_session(
        user_id=user["id"],
        device=device,
        ip_address=ip_address,
        user_agent=user_agent,
        metadata=None
    )
    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user["id"],
        "item_id": None,
        "event_type": "login",
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "duration": None,
        "metadata": {}
    }
    group_id = f"user_{user['id']}"
    send_click_event(event, group_id)
    return {"access_token": access_token, "token_type": "bearer", "session_id": session_id}

@app.post("/api/v1/user/google-register")
async def google_register(request: Request, token: GoogleToken):
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
        user_agent = request.headers.get("user-agent", "")
        ip_address = request.client.host if request.client else ""
        ua = parse(user_agent)
        device = "mobile" if ua.is_mobile else "tablet" if ua.is_tablet else "desktop"
        session_id = await create_session(
            user_id=user["id"],
            device=device,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=None
        )
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user["id"],
            "item_id": None,
            "event_type": "google_register",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {}
        }
        group_id = f"user_{user['id']}"
        send_click_event(event, group_id)
        return {"access_token": access_token, "token_type": "bearer", "session_id": session_id}
    except ValueError as ex:
        print("Google token error:", ex)
        raise HTTPException(status_code=400, detail="Invalid Google token")

@app.post("/api/v1/user/google-login")
async def google_login(request: Request, token: GoogleToken):
    if not token.credential:
        raise HTTPException(status_code=400, detail="Missing Google credential")
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(token.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")

        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=400, detail="User not registered")

        access_token = create_access_token(data={"sub": user["username"]})

        # Get device info
        user_agent = request.headers.get("user-agent", "")
        ip_address = request.client.host if request.client else ""
        ua = parse(user_agent)
        device = "mobile" if ua.is_mobile else "tablet" if ua.is_tablet else "desktop"

        session_id = await create_session(
            user_id=user["id"],
            device=device,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=None
        )
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user["id"],
            "item_id": None,
            "event_type": "google_login",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {}
        }
        group_id = f"user_{user['id']}"
        send_click_event(event, group_id)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "session_id": session_id
        }

    except ValueError as ex:
        print("Google token error:", ex)
        raise HTTPException(status_code=400, detail="Invalid Google token")


@app.post("/api/v1/user/preferences")
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
        session_info = await get_current_active_session_id(user_id)
        session_id = session_info["session_id"] if session_info else None  
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": None,
            "event_type": "preferences_update",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {
                "genres": prefs_in.genres,
                "authors": prefs_in.authors,
                "age": prefs_in.age,
                "pincode": prefs_in.pincode
            }
        }
        group_id = f"user_{user_id}"
        send_click_event(event, group_id)
        create_user_preferences(
            user_id=user_id,
            username=current_user["username"], 
            genres=prefs_in.genres,
            authors=prefs_in.authors,
            age=prefs_in.age,
            pincode=prefs_in.pincode
        )
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
    

@app.get("/api/v1/user/preferences")
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

@app.patch("/api/v1/user/preferences")
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

    session_info = await get_current_active_session_id(user_id)
    session_id = session_info["session_id"] if session_info else None      
    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "item_id": None,
        "event_type": "preferences_update",
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "duration": None,
        "metadata": {
            "genres": genres,
            "authors": authors
        }
    }
    group_id = f"user_{user_id}"
    send_click_event(event, group_id)
    patch_user_preferences(
        user_id=user_id,
        username=current_user["username"],
        old_genres=current.get("genres", []),
        new_genres=genres,
        old_authors=current.get("authors", []),
        new_authors=authors,
        age=prefs_in.age,
        pincode=prefs_in.pincode
    )
    return {
        "user_id": user_id,
        "genres": genres,
        "authors": authors,
        "updated_at": preferences_result.get("updated_at")
    }

@app.get("/api/v1/user/profile")
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

@app.get("/api/v1/user/popular-authors")
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
    

@app.post("/api/v1/user/logout")
async def logout(req: LogoutRequest, current_user=Depends(get_current_user)):
    try:
        # Fetch current active session info (session_id, last_activity)
        session_info = await get_current_active_session_id(current_user["id"])
        if not session_info or session_info["session_id"] != req.session_id:
            # Session mismatch or no active session found
            duration = None
        else:
            last_activity = datetime.fromisoformat(session_info["last_activity"])
            now = datetime.utcnow()
            duration = (now - last_activity).total_seconds()
        
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": current_user["id"],
            "item_id": None,
            "event_type": "logout",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": req.session_id,
            "duration": duration,
            "metadata": {}
        }
        group_id = f"user_{current_user['id']}"
        send_click_event(event, group_id)

        await end_session(req.session_id)
        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {e}")


@app.post("/user/profile_update")
async def save_profile(
    prefs_in: ProfilePreferences, 
    current_user=Depends(get_current_user)
) -> Dict:
    """
    Save  pincode, and age.
    """
    user_id = current_user["id"]
    
    try:
        # Update users table with age and pincode (as part of profile)
        await update_user_profile(
            user_id=user_id,
            age=prefs_in.age,
            pincode=prefs_in.pincode
        ) 
        session_info = await get_current_active_session_id(user_id)
        session_id = session_info["session_id"] if session_info else None   
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": None,
            "event_type": "profile_update",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {
                "age": prefs_in.age,
                "pincode": prefs_in.pincode
            }
        }
        group_id = f"user_{user_id}"
        send_click_event(event, group_id)  
        update_user_profile_fields(
            user_id=user_id,
            age=prefs_in.age,
            pincode=prefs_in.pincode
        )   
        return {
            "user_id": user_id,
            "age": prefs_in.age,
            "pincode": prefs_in.pincode,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid preferences data"
        )
    

@app.get("/api/v1/user/books/{book_id}/reviews-ratings")
async def get_reviews_and_ratings(book_id: str, current_user=Depends(get_current_user)):
    try:
        data = await get_reviews_and_avg_rating_from_db(book_id)
        return data
    except Exception as e:
        print("Error fetching reviews/ratings:", e)
        raise HTTPException(status_code=500, detail="Failed to fetch reviews and ratings")


@app.post("/api/v1/user/books/{book_id}/review")
async def submit_review(
    book_id: str,
    request_body: SubmitReviewRequest,
    current_user=Depends(get_current_user)
):
    user_id = current_user["id"]
    rating = request_body.rating
    content = request_body.text.strip()

    # Validate
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    if not content:
        raise HTTPException(status_code=400, detail="Review content is required")

    try:
        success = await save_review_and_rating_to_db(
            user_id=user_id,
            book_id=book_id,
            rating=rating,
            content=content
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save review")
        
        session_info = await get_current_active_session_id(user_id)
        session_id = session_info["session_id"] if session_info else None   
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": book_id,
            "event_type": "review",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {
                "rating": rating,
                "review_text": content
            }
        }
        group_id = f"user_{user_id}"
        send_click_event(event, group_id)
        create_user_rated_book(
            user_id=user_id,
            book_id=book_id,
            score=rating,
            timestamp=datetime.utcnow().isoformat()
        )

        return {"message": "Review submitted successfully"}
    except Exception as e:
        print("Error saving review:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/user/proxy-epub/{book_id}/")
def proxy_epub(book_id: str):
    epub_url = S3_BUCKET_URL_TEMPLATE.format(book_id=book_id)
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(epub_url, headers=headers, allow_redirects=True, stream=True)
    if r.status_code == 200:
        return Response(content=r.content, media_type="application/epub+zip")
    raise HTTPException(status_code=404, detail="EPUB file not found")

# Serve internal files (e.g. META-INF/container.xml)
@app.get("/api/v1/user/proxy-epub/{book_id}/{internal_path:path}")
def proxy_epub_file(book_id: str, internal_path: str):
    # Download full EPUB from S3
    epub_url = S3_BUCKET_URL_TEMPLATE.format(book_id=book_id)
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(epub_url, headers=headers, allow_redirects=True)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="EPUB not found")

    # Open as ZIP and extract requested file
    epub_io = BytesIO(r.content)
    try:
        with zipfile.ZipFile(epub_io) as zf:
            file_bytes = zf.read(internal_path)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"File {internal_path} not found in EPUB")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read EPUB archive")

    # Guess MIME type
    content_type, _ = mimetypes.guess_type(internal_path)
    content_type = content_type or "application/octet-stream"

    return Response(content=file_bytes, media_type=content_type)


@app.get("/api/v1/user/books/{book_id}/bookmark")
async def get_bookmark_status(book_id: str, current_user=Depends(get_current_user)):
    user_id = current_user["id"]
    try:
        bookmarked = await is_bookmarked_by_user(user_id, book_id)
        return {"bookmarked": bookmarked}
    except Exception as e:
        print("Error checking bookmark:", e)
        raise HTTPException(status_code=500, detail="Failed to check bookmark")

@app.post("/api/v1/user/books/{book_id}/bookmark")
async def add_bookmark(book_id: str, current_user=Depends(get_current_user)):
    user_id = current_user["id"]
    try:
        await add_user_bookmark(user_id, book_id)
        session_info = await get_current_active_session_id(user_id)
        session_id = session_info["session_id"] if session_info else None
 
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": book_id,
            "event_type": "bookmark_add",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {}
        }
        group_id = f"user_{user_id}"
        send_click_event(event, group_id)
        create_user_bookmarked_book(user_id=user_id, book_id=book_id)
        return {"status": "bookmarked"}
    except Exception as e:
        print("Error adding bookmark:", e)
        raise HTTPException(status_code=500, detail="Failed to add bookmark")

@app.delete("/api/v1/user/books/{book_id}/bookmark")
async def remove_bookmark(book_id: str, current_user=Depends(get_current_user)):
    user_id = current_user["id"]
    try:
        await remove_user_bookmark(user_id, book_id)
        session_info = await get_current_active_session_id(user_id)
        session_id = session_info["session_id"] if session_info else None  
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": book_id,
            "event_type": "bookmark_remove",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {}
        }
        group_id = f"user_{user_id}"
        send_click_event(event, group_id)
        delete_user_bookmarked_book(user_id=user_id, book_id=book_id)
        return {"status": "bookmark removed"}
    except Exception as e:
        print("Error removing bookmark:", e)
        raise HTTPException(status_code=500, detail="Failed to remove bookmark")

@app.get("/api/v1/user/bookmarks")
async def get_user_bookmarks(current_user=Depends(get_current_user)):
    user_id = current_user["id"]

    try:
        # Step 1: Get all book IDs the user has bookmarked
        book_ids = await get_user_bookmark_ids(user_id)
        if not book_ids:
            return {"bookmarks": []}

        # Step 2: Fetch full book details
        books = await get_books_by_ids(book_ids)

        # Optional: Preserve order by bookmarked_at (if needed)
        # books are returned in order from get_user_bookmark_ids → get_books_by_ids doesn't guarantee order
        # You can sort here if you stored timestamps (not needed if order isn't critical)

        return {"bookmarks": books}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch data from Supabase")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/user/books/{book_id}/track-read")
async def track_book_read(book_id: str, current_user=Depends(get_current_user)):
    try:
        # Optionally get active session id if you have a helper for that:
        session_info = await get_current_active_session_id(current_user["id"])
        session_id = session_info["session_id"] if session_info else None
        
        event = {
            "event_id": str(uuid.uuid4()),
            "user_id": current_user["id"],
            "item_id": book_id,
            "event_type": "read",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {
                "source": "S3_EPUB",
                "page": "start"
            }
        }
        group_id = f"user_{current_user['id']}"
        send_click_event(event, group_id)
        create_user_read_book(user_id=current_user["id"], book_id=book_id)
        return {"success": True}
    except Exception as e:
        print(f"Track read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to log read event.")


@app.post("/api/v1/user/books/{book_id}/page-turn")
async def page_turn(book_id: str, event: PageTurnEvent, current_user=Depends(get_current_user)):
    try:
        # Optionally get current active session id via helper
        session_info = await get_current_active_session_id(current_user["id"])
        session_id = session_info["session_id"] if session_info else None

        click_event = {
            "event_id": str(uuid.uuid4()),
            "user_id": current_user["id"],
            "item_id": book_id,
            "event_type": "page_turn",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "duration": None,
            "metadata": {
                "page": event.page
            }
        }
        group_id = f"user_{current_user['id']}"
        send_click_event(click_event, group_id)

        return {"success": True}
    except Exception as e:
        print("Error in page-turn event:", e)
        raise HTTPException(status_code=500, detail="Failed to log page-turn event.")
