"""
API routes for DocuMind AI Assistant with PostgreSQL pgvector support
"""

import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.database import get_db
from app.models import Document, DocumentChunk, User, Question, SearchResult
from app.auth import get_current_active_user
from app.services import (
    DocumentProcessor,
    EmbeddingService,
    LLMService,
    QuestionAnsweringService,
)
from app.config import settings

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
document_processor = DocumentProcessor(embedding_service)
llm_service = LLMService()
qa_service = QuestionAnsweringService(embedding_service, llm_service)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Upload and process a document with vector embeddings"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(settings.allowed_extensions)}",
            )

        # Create upload directory if it doesn't exist
        os.makedirs(settings.upload_dir, exist_ok=True)

        # Save file
        file_path = os.path.join(settings.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process and store document with embeddings
        processing_result = document_processor.process_and_store_document(
            file_path, current_user.id, file.filename, db
        )

        if "error" in processing_result:
            # Clean up file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=processing_result["error"])

        return {
            "success": True,
            "document_id": processing_result["document_id"],
            "filename": file.filename,
            "chunks_created": processing_result["chunks_created"],
            "total_chunks": processing_result["total_chunks"],
            "file_size": processing_result["file_info"]["file_size"],
            "status": processing_result["status"],
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Ask a question about uploaded documents using vector search"""
    try:
        # Process question with vector search
        result = await qa_service.ask_question(question, current_user.id, db)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Question processing failed: {str(e)}"
        )


@router.post("/search")
async def search_documents(
    query: str = Form(...),
    top_k: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Search documents using vector similarity"""
    try:
        # Create embedding for search query
        query_embedding = embedding_service.create_embedding(query)

        # Search for similar chunks
        search_results = embedding_service.search_similar(
            db, query_embedding, user_id=current_user.id, top_k=top_k
        )

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                SearchResult(
                    chunk_id=result["chunk_id"],
                    document_id=result["doc_id"],
                    content=result["content"],
                    filename=result["filename"],
                    similarity_score=result["similarity"],
                    chunk_index=result["chunk_index"],
                )
            )

        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_metadata": {
                "embedding_model": settings.embedding_model,
                "similarity_threshold": embedding_service.similarity_threshold,
                "top_k_requested": top_k,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/documents")
async def get_user_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get user's documents with pagination"""
    try:
        # Get documents with pagination
        offset = (page - 1) * page_size
        documents = (
            db.query(Document)
            .filter(Document.user_id == current_user.id)
            .offset(offset)
            .limit(page_size)
            .all()
        )

        # Get total count
        total_count = (
            db.query(Document).filter(Document.user_id == current_user.id).count()
        )

        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
                for doc in documents
            ],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve documents: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a document and its embeddings"""
    try:
        # Get document
        document = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.user_id == current_user.id,
            )
            .first()
        )

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)

        # Delete document chunks and embeddings
        document_processor.delete_document_chunks(document_id, db)

        # Delete document
        db.delete(document)
        db.commit()

        return {"success": True, "message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/questions/history")
async def get_question_history(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get user's question history with sources"""
    try:
        history = qa_service.get_user_question_history(current_user.id, db, limit)
        return {"history": history}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/questions/analytics")
async def get_question_analytics(
    current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)
):
    """Get analytics for user's questions"""
    try:
        analytics = qa_service.get_question_analytics(current_user.id, db)
        return analytics

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve analytics: {str(e)}"
        )


@router.post("/questions/validate")
async def validate_question(question: str = Form(...)):
    """Validate question quality"""
    try:
        validation = qa_service.validate_question_quality(question)
        return validation

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Question validation failed: {str(e)}"
        )


@router.post("/questions/similar")
async def find_similar_questions(
    question: str = Form(...),
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Find similar questions using vector similarity"""
    try:
        similar_questions = qa_service.search_similar_questions(
            question, current_user.id, db, limit
        )
        return {"similar_questions": similar_questions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Similar questions search failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "database": "connected" if settings.database_url else "disconnected",
        "vector_search": "enabled" if settings.use_postgresql else "disabled",
    }


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Get basic stats
        total_documents = db.query(Document).count()
        total_chunks = db.query(DocumentChunk).count()
        total_questions = db.query(Question).count()

        # Get vector search stats
        search_stats = embedding_service.get_search_stats(db)

        return {
            "documents": {
                "total": total_documents,
                "processed": db.query(Document)
                .filter(Document.status == "processed")
                .count(),
                "processing": db.query(Document)
                .filter(Document.status == "processing")
                .count(),
            },
            "chunks": {
                "total": total_chunks,
                "with_embeddings": search_stats.get("chunks_with_embeddings", 0),
                "embedding_coverage": search_stats.get("embedding_coverage", 0.0),
            },
            "questions": {
                "total": total_questions,
            },
            "vector_search": {
                "enabled": settings.use_postgresql,
                "model": settings.embedding_model,
                "dimension": settings.embedding_dimension,
                "similarity_threshold": settings.similarity_threshold,
                "top_k_results": settings.top_k_results,
            },
            "search_stats": search_stats,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve stats: {str(e)}"
        )
