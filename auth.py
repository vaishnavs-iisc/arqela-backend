"""Authentication helpers for Supabase-issued access tokens."""
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import config

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    if not config.SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured")
    return jwt.PyJWKClient(f"{config.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json")


def get_current_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)) -> str:
    """Verify the bearer JWT and return its immutable Supabase user UUID."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Sign in is required.")
    try:
        token = credentials.credentials
        key = _jwks_client().get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256"],
            audience="authenticated",
            issuer=f"{config.SUPABASE_URL.rstrip('/')}/auth/v1",
        )
        return str(claims["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Your session is invalid or has expired.")
