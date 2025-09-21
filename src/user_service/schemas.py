from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: Optional[str] = None

class GenresIn(BaseModel):
    genres: List[str]

class GoogleToken(BaseModel):
    credential: str
