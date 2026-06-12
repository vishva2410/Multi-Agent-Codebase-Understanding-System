# data/sample_code.py
# Sample codebase used for RAG demos — simulates a small auth + user service

import hashlib
import hmac
import os
import time
import jwt  # PyJWT


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key")
ALGORITHM = "HS256"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour


# ──────────────────────────────────────────────
# Password hashing
# ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a plain-text password using SHA-256 + a random salt."""
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain-text password against the stored salt:hash."""
    salt, hashed = stored_hash.split(":")
    return hmac.compare_digest(
        hashlib.sha256(f"{salt}{password}".encode()).hexdigest(),
        hashed
    )


# ──────────────────────────────────────────────
# JWT token creation & validation
# ──────────────────────────────────────────────
def create_jwt_token(user_id: int, email: str) -> str:
    """Create a signed JWT token containing user_id and email."""
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def validate_jwt_token(token: str) -> dict:
    """Validate and decode a JWT token. Raises if expired or invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


# ──────────────────────────────────────────────
# In-memory user store (simulated DB)
# ──────────────────────────────────────────────
_USER_DB: dict[str, dict] = {}


def register_user(email: str, password: str) -> dict:
    """Register a new user. Returns the created user record."""
    if email in _USER_DB:
        raise ValueError(f"User '{email}' already exists.")
    user_id = len(_USER_DB) + 1
    _USER_DB[email] = {
        "id": user_id,
        "email": email,
        "password_hash": hash_password(password),
    }
    return {"id": user_id, "email": email}


def login_user(email: str, password: str) -> str:
    """Authenticate a user and return a JWT token."""
    user = _USER_DB.get(email)
    if not user:
        raise ValueError("User not found.")
    if not verify_password(password, user["password_hash"]):
        raise ValueError("Incorrect password.")
    return create_jwt_token(user["id"], user["email"])


def get_user_profile(token: str) -> dict:
    """Return user profile from a valid JWT token."""
    payload = validate_jwt_token(token)
    email = payload["email"]
    user = _USER_DB.get(email)
    if not user:
        raise ValueError("User no longer exists.")
    return {"id": user["id"], "email": user["email"]}


# ──────────────────────────────────────────────
# Payment processing stub (intentionally minimal)
# ──────────────────────────────────────────────
def process_payment(user_id: int, amount_cents: int, currency: str = "USD") -> dict:
    """
    Stub payment processor. In production this would call Stripe / Razorpay.
    Returns a fake transaction receipt.
    """
    if amount_cents <= 0:
        raise ValueError("Amount must be positive.")
    transaction_id = hashlib.md5(
        f"{user_id}{amount_cents}{time.time()}".encode()
    ).hexdigest()
    return {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "amount_cents": amount_cents,
        "currency": currency,
        "status": "success",
    }
