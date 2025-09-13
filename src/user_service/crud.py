from sqlalchemy.orm import Session
from passlib.context import CryptContext
import models, schemas
from typing import List

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user_with_google(db: Session, user_in: schemas.UserCreate):
    db_user = models.User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=None  
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_preferences_by_user_id(db: Session, user_id: int):
    return db.query(models.UserPreferences).filter(models.UserPreferences.user_id==user_id).first()

def create_or_update_preferences(db: Session, user_id: int, genres: List[str]):
    prefs = db.query(models.UserPreferences).filter(models.UserPreferences.user_id == user_id).first()
    genres_str = ",".join(genres)

    if prefs:  # Update existing preferences
        prefs.genres = genres_str
    else:  # Create new preferences record
        prefs = models.UserPreferences(user_id=user_id, genres=genres_str)
        db.add(prefs)

    try:
        db.commit()
        db.refresh(prefs)
    except Exception as e:
        db.rollback()
        raise e  
    return prefs
