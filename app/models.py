"""
Database models for DocuMind AI Assistant with PostgreSQL pgvector support
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    Boolean,
    Index,
)
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from app.database import Base
from app.config import settings


# SQLAlchemy Models
class User(Base):
    """User model"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="user")


class Document(Base):
    """Document model"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(10), nullable=False)
    status = Column(
        String(20), default="uploaded"
    )  # uploaded, processing, processed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Document chunk model with vector embedding"""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_position = Column(Integer, nullable=False)
    end_position = Column(Integer, nullable=False)
    from pgvector.sqlalchemy import Vector

    embedding = Column(Vector(settings.embedding_dimension), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class Question(Base):
    """Question model for tracking user questions"""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sources = relationship(
        "QuestionSource", back_populates="question", cascade="all, delete-orphan"
    )


class QuestionSource(Base):
    """Question source citations"""

    __tablename__ = "question_sources"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    question = relationship("Question", back_populates="sources")
    chunk = relationship("DocumentChunk")


# Pydantic Models for API
class UserBase(BaseModel):
    """Base user model"""

    username: str
    email: str


class UserCreate(UserBase):
    """User creation model"""

    password: str


class UserResponse(UserBase):
    """User response model"""

    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    """Base document model"""

    filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    """Document creation model"""

    user_id: int


class DocumentResponse(DocumentBase):
    """Document response model"""

    id: int
    user_id: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkBase(BaseModel):
    """Base document chunk model"""

    chunk_index: int
    content: str
    start_position: int
    end_position: int


class DocumentChunkResponse(DocumentChunkBase):
    """Document chunk response model"""

    id: int
    document_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class QuestionBase(BaseModel):
    """Base question model"""

    question_text: str


class QuestionCreate(QuestionBase):
    """Question creation model"""

    user_id: int


class QuestionResponse(QuestionBase):
    """Question response model"""

    id: int
    user_id: int
    answer_text: str
    confidence_score: float
    processing_time: float
    created_at: datetime

    class Config:
        from_attributes = True


class SourceCitation(BaseModel):
    """Source citation model"""

    chunk_id: int
    document_id: int
    similarity_score: float
    content: str
    filename: str


class QuestionAnswerResponse(BaseModel):
    """Question response with sources"""

    question: str
    answer: str
    confidence: float
    sources: List[SourceCitation]
    reasoning: str
    processing_time: float


class SearchResult(BaseModel):
    """Search result model"""

    chunk_id: int
    document_id: int
    content: str
    filename: str
    similarity_score: float
    chunk_index: int
