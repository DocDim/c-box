from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher
from app.config import settings

# bcrypt with an explicit work factor (SDD 5.1: work factor 12)
password_hasher = PasswordHash((BcryptHasher(rounds=settings.BCRYPT_ROUNDS),))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a stored bcrypt hash."""
    return password_hasher.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash from a plain-text password."""
    return password_hasher.hash(password)

def create_access_token(subject: Any, expires_delta: Optional[timedelta] = None) -> str:
    """Mints an HS256 signed JWT with subject id and expiration epoch."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"sub": str(subject), "exp": expire}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
