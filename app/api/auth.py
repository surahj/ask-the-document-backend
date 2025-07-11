"""
Authentication API endpoints for DocuMind AI Assistant
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.models import User, UserResponse
from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
    validate_password,
    validate_email,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.config import settings

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


@router.post("/register", response_model=UserResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    # Validate input
    if not request.username or len(request.username) < 3:
        raise HTTPException(
            status_code=400, detail="Username must be at least 3 characters long"
        )

    if not validate_email(request.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    if not validate_password(request.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long and contain uppercase, lowercase, and numeric characters",
        )

    # Check if username already exists
    stmt = select(User).where(User.username == request.username)
    existing_user = db.execute(stmt).scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Check if email already exists
    stmt = select(User).where(User.email == request.email)
    existing_email = db.execute(stmt).scalar_one_or_none()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(request.password)
    user = User(
        username=request.username,
        email=request.email,
        hashed_password=hashed_password,
        is_active=True,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """Login and get access token"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


@router.post("/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Change user password"""
    # Verify current password
    from app.auth import verify_password

    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Validate new password
    if not validate_password(new_password):
        raise HTTPException(
            status_code=400,
            detail="New password must be at least 8 characters long and contain uppercase, lowercase, and numeric characters",
        )

    # Update password
    current_user.hashed_password = get_password_hash(new_password)

    try:
        db.commit()
        return {"message": "Password changed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Password change failed: {str(e)}")


@router.post("/deactivate")
async def deactivate_account(
    password: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Deactivate user account"""
    # Verify password
    from app.auth import verify_password

    if not verify_password(password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    # Deactivate user
    current_user.is_active = False

    try:
        db.commit()
        return {"message": "Account deactivated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Account deactivation failed: {str(e)}"
        )
