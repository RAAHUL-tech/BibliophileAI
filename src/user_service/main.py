from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.logger import logger
from sqlalchemy.orm import Session
import models, schemas, crud, database, auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from typing import List
from jose import JWTError, jwt
import os

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

ALGORITHM = "HS256"

def get_current_user(token: str = Security(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    user = crud.create_user(db, user)

    # Generate access token for new user (like in google-register)
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, form_data.username)
    if not user or not crud.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


class GoogleToken(BaseModel):
    credential: str

@app.post("/google-register")
def google_register(token: GoogleToken, db: Session = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(token.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        name = idinfo.get("name", email.split("@")[0])
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")
        db_user = crud.get_user_by_email(db, email)
        if db_user:
            raise HTTPException(status_code=400, detail="User already registered")

        user_in = schemas.UserCreate(username=name, email=email, password=None)
        user = crud.create_user_with_google(db, user_in)

        # Create JWT token on registration
        access_token = auth.create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError as ex:
        print("Google token error:", ex)
        raise HTTPException(status_code=400, detail="Invalid Google token")


@app.post("/google-login")
def google_login(token: GoogleToken, db: Session = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(token.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")

        user = crud.get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=400, detail="User not registered")

        access_token = auth.create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google token")
    

class GenresIn(BaseModel):
    genres: List[str]


@app.post("/user/preferences", response_model=schemas.UserPreferencesResponse)
def save_preferences(prefs_in: GenresIn, db: Session=Depends(get_db), current_user: models.User=Depends(get_current_user)):
    prefs = crud.create_or_update_preferences(db, current_user.id, prefs_in.genres)
    genres_list = prefs.genres.split(",") if prefs.genres else []
    return schemas.UserPreferencesResponse(
        id=prefs.id,
        user_id=current_user.id,
        genres=genres_list
    )


@app.get("/user/preferences", response_model=schemas.UserPreferencesResponse)
def get_preferences(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = crud.get_preferences_by_user_id(db, current_user.id)
    if not prefs:
        return schemas.UserPreferencesResponse(id=0, user_id=current_user.id, genres=[])
    genres_list = prefs.genres.split(",") if prefs.genres else []
    return schemas.UserPreferencesResponse(
        id=prefs.id,
        user_id=current_user.id,
        genres=genres_list
    )

@app.get("/user/profile")
def get_user_profile(current_user: models.User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email
    }
