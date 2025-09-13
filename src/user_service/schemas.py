from pydantic import BaseModel
from typing import List, Optional

class UserCreate(BaseModel):
    username: str
    email: str
    password: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        orm_mode = True


class UserPreferencesBase(BaseModel):
    genres: List[str]

class UserPreferencesCreate(UserPreferencesBase):
    pass

class UserPreferencesResponse(UserPreferencesBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
