"""
Authentication API endpoints for DocuMind AI Assistant
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.models import User, UserResponse, OTP
from app.services.email_service import EmailService
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
import random
from datetime import datetime, timedelta

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
    """Register a new user (requires verified OTP)"""
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

    # Enforce OTP verification for signup
    otp = (
        db.query(OTP)
        .filter(
            OTP.email == request.email,
            OTP.purpose == "signup",
            OTP.is_used == False,
            OTP.expires_at > datetime.utcnow(),
        )
        .order_by(OTP.created_at.desc())
        .first()
    )
    if not otp:
        raise HTTPException(
            status_code=400,
            detail="OTP verification required. Please verify your email with the OTP sent.",
        )

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
        # Mark OTP as used only after successful registration
        otp.is_used = True
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


# Utility to generate a 6-digit OTP


def generate_otp():
    return str(random.randint(100000, 999999))


# Endpoint: Send OTP (for signup or password reset)
@router.post("/send-otp")
async def send_otp(
    email: str = Body(...),
    purpose: str = Body(..., regex="^(signup|reset)$"),
    username: str = Body(None),
    db: Session = Depends(get_db),
):
    """Send OTP for signup or password reset"""
    user = db.query(User).filter(User.email == email).first()
    if purpose == "signup":
        if user:
            raise HTTPException(
                status_code=400,
                detail="This email is already registered. Please log in or use 'Forgot Password' to reset your password.",
            )
        if username:
            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail="This username is already taken. Please choose another username.",
                )
    if purpose == "reset" and not user:
        raise HTTPException(status_code=404, detail="Email not found")
    first_name = user.username if user else email.split("@")[0]
    otp_code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    # Invalidate previous OTPs for this email/purpose
    db.query(OTP).filter(
        OTP.email == email, OTP.purpose == purpose, OTP.is_used == False
    ).update({"is_used": True})
    otp = OTP(
        email=email,
        otp_code=otp_code,
        purpose=purpose,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
        is_used=False,
    )
    db.add(otp)
    db.commit()
    db.refresh(otp)
    email_service = EmailService()
    if purpose == "signup":
        await email_service.send_otp_email(email, first_name, otp_code)
    else:
        await email_service.send_password_reset_email(email, first_name, otp_code)
    return {"message": f"OTP sent to {email}"}


# Endpoint: Verify OTP
@router.post("/verify-otp")
async def verify_otp(
    email: str = Body(...),
    otp_code: str = Body(...),
    purpose: str = Body(..., regex="^(signup|reset)$"),
    db: Session = Depends(get_db),
):
    """Verify OTP for signup or password reset (does NOT mark as used)"""
    otp = (
        db.query(OTP)
        .filter(
            OTP.email == email,
            OTP.otp_code == otp_code,
            OTP.purpose == purpose,
            OTP.is_used == False,
            OTP.expires_at > datetime.utcnow(),
        )
        .first()
    )
    if not otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    # Do NOT mark as used here
    return {"message": "OTP verified"}


# Endpoint: Reset password after OTP verification
@router.post("/reset-password")
async def reset_password(
    email: str = Body(...),
    otp_code: str = Body(...),
    new_password: str = Body(...),
    db: Session = Depends(get_db),
):
    """Reset password after OTP verification"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    otp = (
        db.query(OTP)
        .filter(
            OTP.email == email,
            OTP.otp_code == otp_code,
            OTP.purpose == "reset",
            OTP.is_used == False,
            OTP.expires_at > datetime.utcnow(),
        )
        .first()
    )
    if not otp:
        raise HTTPException(status_code=400, detail="OTP not verified or expired")
    if not validate_password(new_password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long and contain uppercase, lowercase, and numeric characters",
        )
    user.hashed_password = get_password_hash(new_password)
    # Mark OTP as used only after successful password reset
    otp.is_used = True
    db.commit()
    return {"message": "Password reset successful"}
