from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict, Field

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    # bcrypt only uses the first 72 bytes of a password
    password: str = Field(min_length=8, max_length=72)

class UserResponse(UserBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: Optional[str] = None