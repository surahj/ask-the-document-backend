"""
Background task service for DocuMind AI Assistant
"""

import asyncio
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """Service for handling background tasks like document processing and embedding creation"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.document_processor = DocumentProcessor(self.embedding_service)

    async def process_document_async(
        self,
        document_id: int,
        file_content: bytes,
        file_ext: str,
        filename: str,
        user_id: int,
        file_size: int,
        cloudinary_url: str = None,
        cloudinary_public_id: str = None,
    ):
        """Process document completely asynchronously (text extraction + chunking + embeddings)"""
        logger.info(f"Starting async document processing for document {document_id}")

        try:
            # Get a new database session for this background task
            db = SessionLocal()

            try:
                # Query for the existing document record instead of creating a new one
                from app.models import Document

                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    logger.error(f"Document {document_id} not found in database")
                    return

                # Update document status to processing
                document.status = "processing"
                db.commit()
                logger.info(f"Updated document {document_id} status to processing")

                # Process document in thread pool (text extraction + chunking)
                logger.info(
                    f"Starting text extraction and chunking for document {document_id}"
                )
                loop = asyncio.get_event_loop()

                # Run document processing in thread pool to avoid blocking
                processing_result = await loop.run_in_executor(
                    None, self._process_document_sync, file_content, file_ext, filename
                )

                if "error" in processing_result:
                    logger.error(
                        f"Document processing failed for {document_id}: {processing_result['error']}"
                    )
                    document.status = "error"
                    db.commit()
                    return

                chunks = processing_result.get("chunks", [])
                logger.info(
                    f"Successfully processed document {document_id} into {len(chunks)} chunks"
                )

                # Create all chunk records first (without embeddings)
                from app.models import DocumentChunk

                chunk_records = []
                for chunk_data in chunks:
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_data["chunk_index"],
                        content=chunk_data["content"],
                        start_position=chunk_data["start_position"],
                        end_position=chunk_data["end_position"],
                    )
                    db.add(chunk)
                    chunk_records.append(chunk)

                # Flush to get all chunk IDs
                db.flush()
                logger.info(
                    f"Created {len(chunk_records)} chunk records for document {document_id}"
                )

                # Extract all chunk contents for batch embedding creation
                chunk_contents = [chunk_data["content"] for chunk_data in chunks]

                # Create embeddings in batch - run in thread pool to avoid blocking
                logger.info(
                    f"Creating batch embeddings for {len(chunk_contents)} chunks"
                )

                # Run embedding creation in a thread pool to avoid blocking the event loop
                embeddings = await loop.run_in_executor(
                    None, self.embedding_service.batch_create_embeddings, chunk_contents
                )

                # Assign embeddings to chunks
                successful_embeddings = 0
                for i, (chunk, embedding) in enumerate(zip(chunk_records, embeddings)):
                    if embedding:
                        chunk.embedding = embedding
                        successful_embeddings += 1
                    else:
                        logger.warning(
                            f"Failed to create embedding for chunk {chunks[i]['chunk_index']}"
                        )

                # Update document status to processed
                document.status = "processed"
                db.commit()

                logger.info(
                    f"Successfully created {successful_embeddings}/{len(chunks)} embeddings for document {document_id}"
                )
                logger.info(f"Updated document {document_id} status to 'processed'")

            except Exception as e:
                # Update document status to error
                try:
                    document = (
                        db.query(Document).filter(Document.id == document_id).first()
                    )
                    if document:
                        document.status = "error"
                        db.commit()
                        logger.error(
                            f"Updated document {document_id} status to 'error'"
                        )
                except:
                    pass

                logger.error(
                    f"Error in async document processing for document {document_id}: {e}"
                )
                raise

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")

    def _process_document_sync(
        self, file_content: bytes, file_ext: str, filename: str
    ) -> Dict[str, Any]:
        """Synchronous document processing (text extraction + chunking) - runs in thread pool"""
        try:
            # Save to a temporary file for processing
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            try:
                # Validate document
                validation = self.document_processor.validate_document(temp_file_path)
                if "error" in validation:
                    return validation

                # Extract text
                extraction = self.document_processor.extract_text(temp_file_path)
                if "error" in extraction:
                    return extraction

                # Chunk text
                chunks = self.document_processor.chunk_text(extraction["text"])
                if not chunks or (isinstance(chunks, list) and "error" in chunks[0]):
                    return {"error": "Failed to chunk document"}

                return {"chunks": chunks}
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        except Exception as e:
            return {"error": f"Document processing error: {str(e)}"}

    async def create_embeddings_async(
        self, document_id: int, chunks: List[Dict[str, Any]]
    ):
        """Create embeddings for document chunks asynchronously (legacy method for backward compatibility)"""
        logger.info(f"Starting async embedding creation for document {document_id}")

        try:
            # Get a new database session for this background task
            db = SessionLocal()

            try:
                # Update document status to processing
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    logger.error(f"Document {document_id} not found")
                    return

                document.status = "processing"
                db.commit()
                logger.info(f"Updated document {document_id} status to 'processing'")

                # Create all chunk records first (without embeddings)
                chunk_records = []
                for chunk_data in chunks:
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_data["chunk_index"],
                        content=chunk_data["content"],
                        start_position=chunk_data["start_position"],
                        end_position=chunk_data["end_position"],
                    )
                    db.add(chunk)
                    chunk_records.append(chunk)

                # Flush to get all chunk IDs
                db.flush()
                logger.info(
                    f"Created {len(chunk_records)} chunk records for document {document_id}"
                )

                # Extract all chunk contents for batch embedding creation
                chunk_contents = [chunk_data["content"] for chunk_data in chunks]

                # Create embeddings in batch - run in thread pool to avoid blocking
                logger.info(
                    f"Creating batch embeddings for {len(chunk_contents)} chunks"
                )

                # Run embedding creation in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, self.embedding_service.batch_create_embeddings, chunk_contents
                )

                # Assign embeddings to chunks
                successful_embeddings = 0
                for i, (chunk, embedding) in enumerate(zip(chunk_records, embeddings)):
                    if embedding:
                        chunk.embedding = embedding
                        successful_embeddings += 1
                    else:
                        logger.warning(
                            f"Failed to create embedding for chunk {chunks[i]['chunk_index']}"
                        )

                # Update document status to processed
                document.status = "processed"
                db.commit()

                logger.info(
                    f"Successfully created {successful_embeddings}/{len(chunks)} embeddings for document {document_id}"
                )
                logger.info(f"Updated document {document_id} status to 'processed'")

            except Exception as e:
                # Update document status to error
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.status = "error"
                    db.commit()
                    logger.error(f"Updated document {document_id} status to 'error'")

                logger.error(
                    f"Error in async embedding creation for document {document_id}: {e}"
                )
                raise

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to create embeddings for document {document_id}: {e}")


# Global instance
background_task_service = BackgroundTaskService()


async def process_document_background_task(
    document_id: int,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: int,
    file_size: int,
    cloudinary_url: str = None,
    cloudinary_public_id: str = None,
):
    """Background task wrapper for complete document processing"""
    # Create a new task that runs independently
    task = asyncio.create_task(
        background_task_service.process_document_async(
            document_id,
            file_content,
            file_ext,
            filename,
            user_id,
            file_size,
            cloudinary_url,
            cloudinary_public_id,
        )
    )

    # Don't await the task - let it run independently
    logger.info(
        f"Created independent background task for document processing {document_id}"
    )

    # Optional: Add error handling for the task
    def handle_task_exception(task):
        try:
            task.result()
        except Exception as e:
            logger.error(f"Background task for document {document_id} failed: {e}")

    task.add_done_callback(handle_task_exception)


async def create_embeddings_background_task(
    document_id: int, chunks: List[Dict[str, Any]]
):
    """Background task wrapper for embedding creation (legacy method)"""
    # Create a new task that runs independently
    task = asyncio.create_task(
        background_task_service.create_embeddings_async(document_id, chunks)
    )

    # Don't await the task - let it run independently
    logger.info(f"Created independent background task for document {document_id}")

    # Optional: Add error handling for the task
    def handle_task_exception(task):
        try:
            task.result()
        except Exception as e:
            logger.error(f"Background task for document {document_id} failed: {e}")

    task.add_done_callback(handle_task_exception)
