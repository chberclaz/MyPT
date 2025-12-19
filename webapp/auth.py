"""
MyPT Authentication Module

Simple but secure authentication with JWT tokens and role-based access.

Roles:
    - user: Can access Chat/RAG interface
    - admin: Can access Chat AND Training pages

Usage:
    # In routes, use dependencies:
    @router.get("/protected")
    async def protected_route(user: User = Depends(require_user)):
        ...
    
    @router.get("/admin-only")
    async def admin_route(user: User = Depends(require_admin)):
        ...
"""

import os
import json
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict

from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import RedirectResponse

# Use passlib for secure password hashing
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    # Fallback to hashlib if passlib not available (less secure but works offline)
    import hashlib
    class SimplePwdContext:
        def hash(self, password: str) -> str:
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return f"{salt}${hashed.hex()}"
        
        def verify(self, password: str, hashed: str) -> bool:
            try:
                salt, hash_hex = hashed.split('$')
                expected = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
                return secrets.compare_digest(expected.hex(), hash_hex)
            except Exception:
                return False
    pwd_context = SimplePwdContext()

# Use python-jose for JWT, fallback to simple signed tokens
try:
    from jose import jwt, JWTError
    HAVE_JOSE = True
except ImportError:
    import hmac
    import base64
    HAVE_JOSE = False


# ============================================================================
# Configuration
# ============================================================================

WEBAPP_DIR = Path(__file__).parent
USERS_FILE = WEBAPP_DIR / "users.json"

# JWT settings
SECRET_KEY = os.environ.get("MYPT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
COOKIE_NAME = "mypt_session"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class User:
    username: str
    role: str  # "user" or "admin"
    
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass  
class UserInDB(User):
    hashed_password: str


# ============================================================================
# Token Functions
# ============================================================================

def create_token(username: str, role: str) -> str:
    """Create a JWT token for the user."""
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire.timestamp()
    }
    
    if HAVE_JOSE:
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    else:
        # Simple fallback: base64(payload) + HMAC signature
        payload_bytes = json.dumps(payload).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode()
        signature = hmac.new(SECRET_KEY.encode(), payload_bytes, 'sha256').hexdigest()
        return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[User]:
    """Verify a JWT token and return the user."""
    try:
        if HAVE_JOSE:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        else:
            # Verify simple token
            payload_b64, signature = token.rsplit('.', 1)
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            expected_sig = hmac.new(SECRET_KEY.encode(), payload_bytes, 'sha256').hexdigest()
            if not secrets.compare_digest(signature, expected_sig):
                return None
            payload = json.loads(payload_bytes)
        
        # Check expiration
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            return None
        
        username = payload.get("sub")
        role = payload.get("role", "user")
        
        if username:
            return User(username=username, role=role)
        return None
        
    except Exception:
        return None


# ============================================================================
# User Management
# ============================================================================

def load_users() -> dict:
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        # Create default users file
        create_default_users()
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: dict):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def create_default_users():
    """Create default admin and user accounts."""
    default_users = {
        "admin": {
            "hashed_password": pwd_context.hash("admin"),  # Change in production!
            "role": "admin"
        },
        "user": {
            "hashed_password": pwd_context.hash("user"),  # Change in production!
            "role": "user"
        }
    }
    save_users(default_users)
    print(f"Created default users file: {USERS_FILE}")
    print("  ⚠️  Default credentials - CHANGE IN PRODUCTION:")
    print("     admin:admin (admin role)")
    print("     user:user (user role)")


def get_user(username: str) -> Optional[UserInDB]:
    """Get a user from the database."""
    users = load_users()
    if username in users:
        user_data = users[username]
        return UserInDB(
            username=username,
            role=user_data.get("role", "user"),
            hashed_password=user_data.get("hashed_password", "")
        )
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not pwd_context.verify(password, user.hashed_password):
        return None
    return User(username=user.username, role=user.role)


def create_user(username: str, password: str, role: str = "user") -> bool:
    """Create a new user."""
    users = load_users()
    if username in users:
        return False
    
    users[username] = {
        "hashed_password": pwd_context.hash(password),
        "role": role
    }
    save_users(users)
    return True


def change_password(username: str, new_password: str) -> bool:
    """Change a user's password."""
    users = load_users()
    if username not in users:
        return False
    
    users[username]["hashed_password"] = pwd_context.hash(new_password)
    save_users(users)
    return True


# ============================================================================
# FastAPI Dependencies
# ============================================================================

async def get_current_user(request: Request) -> Optional[User]:
    """Get the current user from the session cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return verify_token(token)


async def require_user(request: Request) -> User:
    """Dependency that requires any authenticated user."""
    user = await get_current_user(request)
    if not user:
        # For API endpoints, raise 401
        if request.url.path.startswith("/api/"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        # For pages, redirect to login
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": f"/login?next={request.url.path}"}
        )
    return user


async def require_admin(request: Request) -> User:
    """Dependency that requires admin role."""
    user = await require_user(request)
    if not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


async def optional_user(request: Request) -> Optional[User]:
    """Dependency that optionally gets the current user (no error if not logged in)."""
    return await get_current_user(request)


# ============================================================================
# Response Helpers
# ============================================================================

def create_login_response(user: User, redirect_url: str = "/") -> RedirectResponse:
    """Create a response that logs in the user."""
    token = create_token(user.username, user.role)
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,  # Not accessible via JavaScript
        secure=False,   # Set to True in production with HTTPS
        samesite="lax",
        max_age=TOKEN_EXPIRE_HOURS * 3600
    )
    return response


def create_logout_response(redirect_url: str = "/login") -> RedirectResponse:
    """Create a response that logs out the user."""
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key=COOKIE_NAME)
    return response

