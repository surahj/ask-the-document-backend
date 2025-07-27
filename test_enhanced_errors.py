#!/usr/bin/env python3
"""
Test script to verify enhanced error messages for processing documents
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.qa_service import QuestionAnsweringService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_error_messages():
    """Test that enhanced error messages are returned for processing documents"""

    # Initialize services
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    qa_service = QuestionAnsweringService(embedding_service, llm_service)

    # Create test scenarios
    db = SessionLocal()
    try:
        # Test 1: Document in "uploaded" status
        logger.info("Testing document in 'uploaded' status...")
        uploaded_doc = Document(
            user_id=1,
            filename="test_uploaded.pdf",
            file_size=1024000,
            file_type=".pdf",
            status="uploaded",
        )
        db.add(uploaded_doc)
        db.commit()

        # Test asking a question with uploaded document
        result = await qa_service.ask_question("What is this document about?", 1, db)

        logger.info("Uploaded document error response:")
        logger.info(f"Error: {result.get('error', 'No error')}")
        logger.info(f"Processing status: {result.get('processing_status', 'None')}")
        logger.info(f"Processing phase: {result.get('processing_phase', 'None')}")
        logger.info(f"Progress info: {result.get('progress_info', 'None')}")
        logger.info(f"Estimated time: {result.get('estimated_time', 'None')}")

        # Clean up
        db.delete(uploaded_doc)
        db.commit()

        # Test 2: Document in "processing" status with chunks
        logger.info("\nTesting document in 'processing' status with chunks...")
        processing_doc = Document(
            user_id=1,
            filename="test_processing.pdf",
            file_size=1024000,
            file_type=".pdf",
            status="processing",
        )
        db.add(processing_doc)
        db.flush()

        # Add some chunks without embeddings
        for i in range(5):
            chunk = DocumentChunk(
                document_id=processing_doc.id,
                chunk_index=i,
                content=f"This is chunk {i} content.",
                start_position=i * 100,
                end_position=(i + 1) * 100,
            )
            db.add(chunk)

        # Add some chunks with embeddings
        for i in range(3):
            chunk = DocumentChunk(
                document_id=processing_doc.id,
                chunk_index=i + 5,
                content=f"This is chunk {i + 5} content with embedding.",
                start_position=(i + 5) * 100,
                end_position=(i + 6) * 100,
                embedding=[0.1] * 384,  # Mock embedding
            )
            db.add(chunk)

        db.commit()

        # Test asking a question with processing document
        result = await qa_service.ask_question("What is this document about?", 1, db)

        logger.info("Processing document error response:")
        logger.info(f"Error: {result.get('error', 'No error')}")
        logger.info(f"Processing status: {result.get('processing_status', 'None')}")
        logger.info(f"Processing phase: {result.get('processing_phase', 'None')}")
        logger.info(f"Progress info: {result.get('progress_info', 'None')}")
        logger.info(
            f"Chunks with embeddings: {result.get('chunks_with_embeddings', 'None')}"
        )
        logger.info(f"Total chunks: {result.get('total_chunks', 'None')}")
        logger.info(f"Progress percentage: {result.get('progress_percentage', 'None')}")
        logger.info(f"Estimated time: {result.get('estimated_time', 'None')}")

        # Clean up
        db.delete(processing_doc)
        db.commit()

        # Test 3: Document in "error" status
        logger.info("\nTesting document in 'error' status...")
        error_doc = Document(
            user_id=1,
            filename="test_error.pdf",
            file_size=1024000,
            file_type=".pdf",
            status="error",
        )
        db.add(error_doc)
        db.commit()

        # Test asking a question with error document
        result = await qa_service.ask_question("What is this document about?", 1, db)

        logger.info("Error document error response:")
        logger.info(f"Error: {result.get('error', 'No error')}")
        logger.info(f"Processing status: {result.get('processing_status', 'None')}")
        logger.info(f"Suggestions: {result.get('suggestions', 'None')}")

        # Clean up
        db.delete(error_doc)
        db.commit()

        logger.info("\n✅ All enhanced error message tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(test_enhanced_error_messages())
