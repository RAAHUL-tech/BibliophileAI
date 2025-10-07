from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: Optional[str] = None

class GoogleToken(BaseModel):
    credential: str

class UserPreferences(BaseModel):
    genres: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    pincode: Optional[str] = None
    age: Optional[int] = None

