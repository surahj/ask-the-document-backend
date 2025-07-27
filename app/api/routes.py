"""
API routes for DocuMind AI Assistant with PostgreSQL pgvector support
"""

import os
import shutil
import asyncio
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
)
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
from app.services.cloudinary_service import CloudinaryService
from app.services.background_tasks import process_document_background_task
from app.config import settings
import math
import pprint
import logging

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
document_processor = DocumentProcessor(embedding_service)
llm_service = LLMService()
qa_service = QuestionAnsweringService(embedding_service, llm_service)
cloudinary_service = CloudinaryService()


def sanitize_json(obj):
    # Handle Pydantic/BaseModel objects
    if hasattr(obj, "dict"):
        return sanitize_json(obj.dict())
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else 0.0
    else:
        return obj


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    try:
        logging.info(
            f"Received upload: filename={file.filename}, content_type={file.content_type}"
        )

        # Check if user already has a document
        existing_documents = (
            db.query(Document).filter(Document.user_id == current_user.id).all()
        )

        if existing_documents:
            existing_doc = existing_documents[0]  # Get the first document
            return {
                "success": False,
                "error": "DOCUMENT_EXISTS",
                "message": f"You already have a document uploaded: '{existing_doc.filename}'. Please delete it first before uploading a new document.",
                "existing_document": {
                    "id": existing_doc.id,
                    "filename": existing_doc.filename,
                    "file_type": existing_doc.file_type,
                    "file_size": existing_doc.file_size,
                    "created_at": existing_doc.created_at.isoformat(),
                },
            }

        # Read file content ONCE
        file_content = await file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        logging.info(f"File extension: {file_ext}, size={len(file_content)} bytes")

        # Upload to Cloudinary for storage/reference
        cloudinary_url = None
        cloudinary_public_id = None
        if settings.use_cloudinary and cloudinary_service.is_available():
            logging.info("Uploading to Cloudinary for storage/reference.")
            upload_result = cloudinary_service.upload_file(
                file_content, file.filename, file_ext
            )
            logging.info(f"Cloudinary upload_result: {upload_result}")
            if "error" in upload_result:
                logging.error(f"Cloudinary upload error: {upload_result['error']}")
                raise HTTPException(status_code=500, detail=upload_result["error"])
            cloudinary_url = upload_result["url"]
            cloudinary_public_id = upload_result["public_id"]

        # Save document record immediately (without processing)
        logging.info(
            "Saving document record to database (processing will happen in background)..."
        )
        document_id = document_processor.save_document_record_async(
            user_id=current_user.id,
            filename=file.filename,
            file_size=len(file_content),
            file_type=file_ext,
            db=db,
            cloudinary_url=cloudinary_url,
            cloudinary_public_id=cloudinary_public_id,
            chunks=[],  # No chunks yet - will be created in background
        )

        # Ensure document is committed before starting background task
        await asyncio.sleep(0.1)

        # Create truly async background task for complete document processing
        logging.info(
            f"Starting background document processing for document {document_id}"
        )
        task = asyncio.create_task(
            process_document_background_task(
                document_id,
                file_content,
                file_ext,
                file.filename,
                current_user.id,
                len(file_content),
                cloudinary_url,
                cloudinary_public_id,
            )
        )
        logging.info(
            f"Created truly async background task for complete document processing (document {document_id})"
        )

        logging.info(
            f"Document upload completed successfully. Document ID: {document_id} (processing in background)"
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": 0,  # Will be updated as processing progresses
            "total_chunks": 0,  # Will be updated as processing progresses
            "file_size": len(file_content),
            "status": "uploaded",  # Document is uploaded but processing is still happening
            "processing_status": "document_processing",  # New field to indicate background processing
            "storage_type": (
                "cloudinary"
                if settings.use_cloudinary and cloudinary_service.is_available()
                else "local"
            ),
            "cloudinary_url": cloudinary_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/replace")
async def replace_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Replace existing document with a new one"""
    try:
        logging.info(
            f"Received document replacement: filename={file.filename}, content_type={file.content_type}"
        )

        # Check if user has an existing document
        existing_documents = (
            db.query(Document).filter(Document.user_id == current_user.id).all()
        )

        if not existing_documents:
            raise HTTPException(
                status_code=400,
                detail="No existing document found to replace. Please use the regular upload endpoint.",
            )

        existing_doc = existing_documents[0]
        logging.info(f"Replacing existing document: {existing_doc.filename}")

        # Delete existing document and its chunks
        logging.info("Deleting existing document chunks...")
        success = document_processor.delete_document_chunks(existing_doc.id, db)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to delete existing document chunks"
            )

        # Delete the document record
        db.delete(existing_doc)
        db.commit()

        logging.info(f"Successfully deleted existing document: {existing_doc.filename}")

        # Now proceed with the new upload
        file_content = await file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        logging.info(f"File extension: {file_ext}, size={len(file_content)} bytes")

        # Upload to Cloudinary for storage/reference
        cloudinary_url = None
        cloudinary_public_id = None
        if settings.use_cloudinary and cloudinary_service.is_available():
            logging.info("Uploading to Cloudinary for storage/reference.")
            upload_result = cloudinary_service.upload_file(
                file_content, file.filename, file_ext
            )
            logging.info(f"Cloudinary upload_result: {upload_result}")
            if "error" in upload_result:
                logging.error(f"Cloudinary upload error: {upload_result['error']}")
                raise HTTPException(status_code=500, detail=upload_result["error"])
            cloudinary_url = upload_result["url"]
            cloudinary_public_id = upload_result["public_id"]

        # Save new document record immediately (without processing)
        logging.info(
            "Saving new document record to database (processing will happen in background)..."
        )
        document_id = document_processor.save_document_record_async(
            user_id=current_user.id,
            filename=file.filename,
            file_size=len(file_content),
            file_type=file_ext,
            db=db,
            cloudinary_url=cloudinary_url,
            cloudinary_public_id=cloudinary_public_id,
            chunks=[],  # No chunks yet - will be created in background
        )

        # Ensure document is committed before starting background task
        await asyncio.sleep(0.1)

        # Create truly async background task for complete document processing
        logging.info(
            f"Starting background document processing for document {document_id}"
        )
        task = asyncio.create_task(
            process_document_background_task(
                document_id,
                file_content,
                file_ext,
                file.filename,
                current_user.id,
                len(file_content),
                cloudinary_url,
                cloudinary_public_id,
            )
        )
        logging.info(
            f"Created truly async background task for complete document processing (document {document_id})"
        )

        logging.info(
            f"Document replacement completed successfully. Document ID: {document_id} (processing in background)"
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": 0,  # Will be updated as processing progresses
            "total_chunks": 0,  # Will be updated as processing progresses
            "file_size": len(file_content),
            "status": "uploaded",  # Document is uploaded but processing is still happening
            "processing_status": "document_processing",  # New field to indicate background processing
            "storage_type": (
                "cloudinary"
                if settings.use_cloudinary and cloudinary_service.is_available()
                else "local"
            ),
            "cloudinary_url": cloudinary_url,
            "replaced_document": {
                "id": existing_doc.id,
                "filename": existing_doc.filename,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Document replacement failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Document replacement failed: {str(e)}"
        )


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

        sanitized = sanitize_json(result)
        return sanitized

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

        # Delete file from storage
        if document.cloudinary_public_id:
            # Delete from Cloudinary
            delete_result = cloudinary_service.delete_file(
                document.cloudinary_public_id
            )
            if "error" in delete_result:
                # Log error but continue with database cleanup
                print(f"Failed to delete Cloudinary file: {delete_result['error']}")
        elif document.file_path and os.path.exists(document.file_path):
            # Delete local file
            os.remove(document.file_path)

        # Delete document chunks and embeddings
        success = document_processor.delete_document_chunks(document_id, db)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to delete document chunks"
            )

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
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get user's question history with pagination"""
    try:
        result = qa_service.get_user_question_history(
            current_user.id, db, page=page, page_size=page_size
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve history: {str(e)}"
        )


@router.delete("/questions/{question_id}")
async def delete_question(
    question_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a question by ID (only for the current user)"""
    question = (
        db.query(Question)
        .filter(Question.id == question_id, Question.user_id == current_user.id)
        .first()
    )
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    db.delete(question)
    db.commit()
    return {"message": "Question deleted successfully"}


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


@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get document processing status"""
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

        # Count chunks with embeddings
        chunks_with_embeddings = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.isnot(None),
            )
            .count()
        )

        total_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .count()
        )

        # Determine processing status
        if document.status == "uploaded":
            processing_status = "document_processing"
            message = f"Document uploaded successfully. Processing document (text extraction and chunking)..."
        elif document.status == "processing":
            if total_chunks == 0:
                processing_status = "document_processing"
                message = f"Processing document (text extraction and chunking)..."
            else:
                processing_status = "embeddings_processing"
                message = f"Creating embeddings... ({chunks_with_embeddings}/{total_chunks} completed)"
        elif document.status == "processed":
            processing_status = "ready"
            message = "Document is ready for questions!"
        elif document.status == "error":
            processing_status = "error"
            message = "Error occurred during processing. Please try uploading again."
        else:
            processing_status = "unknown"
            message = "Unknown processing status"

        return {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "processing_status": processing_status,
            "message": message,
            "chunks_with_embeddings": chunks_with_embeddings,
            "total_chunks": total_chunks,
            "progress_percentage": (
                (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
            ),
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting document status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get document status: {str(e)}"
        )
