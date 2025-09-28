from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
import os
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from schemas import UserCreate, GenresIn, GoogleToken
from supabase_client import (
    get_user_by_username,
    get_user_by_email,
    create_user,
    create_or_update_preferences,
    get_preferences_by_user_id
)
from auth import create_access_token, decode_access_token
from user_embeddings import create_user_embedding_vectors

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_BCRYPT_BYTES = 72

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

def truncate_password(password: str) -> str:
    encoded = password.encode('utf-8')[:MAX_BCRYPT_BYTES]
    return encoded.decode('utf-8', errors='ignore')

@app.post("/register")
async def register(user: UserCreate):
    existing_user = await get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    safe_password = truncate_password(user.password)
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
    safe_password = truncate_password(form_data.password)
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
async def save_preferences(prefs_in: GenresIn, current_user=Depends(get_current_user)):
    prefs = await create_or_update_preferences(current_user["id"], prefs_in.genres)
    
    if prefs is None:
        genres_list = list(prefs_in.genres) if isinstance(prefs_in.genres, (list, tuple)) else []
    else:
        genres_list = prefs.get("genres", "").split(",") if prefs.get("genres") else []
    
    try:
        await create_user_embedding_vectors(current_user["id"], genres_list)
    except Exception as e:
        print(f"Error creating embedding for user {current_user['id']}: {e}")
    
    if not prefs:
        return {"id": current_user["id"], "user_id": current_user["id"], "genres": genres_list}
    else:
        return {"id": prefs["id"], "user_id": current_user["id"], "genres": genres_list}

@app.get("/user/preferences")
async def get_preferences(current_user=Depends(get_current_user)):
    prefs = await get_preferences_by_user_id(current_user["id"])
    if not prefs:
        return {"id": 0, "user_id": current_user["id"], "genres": []}
    genres_list = prefs.get("genres", "").split(",") if prefs.get("genres") else []
    return {"id": prefs["id"], "user_id": current_user["id"], "genres": genres_list}

@app.get("/user/profile")
async def get_user_profile(current_user=Depends(get_current_user)):
    return {"username": current_user["username"], "email": current_user["email"]}
