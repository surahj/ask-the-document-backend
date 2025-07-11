"""
Document processing service for DocuMind AI Assistant with vector embeddings
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
import markdown
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Document, DocumentChunk
from app.services.embedding_service import EmbeddingService


class DocumentProcessor:
    """Document processing service with vector embedding support"""

    def __init__(self, embedding_service: EmbeddingService = None):
        self.supported_extensions = settings.allowed_extensions
        self.max_file_size = settings.max_file_size
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_chunks = settings.max_chunks_per_document
        self.embedding_service = embedding_service or EmbeddingService()

    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document file"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": "File not found"}

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {
                    "error": f"File size exceeds maximum limit of {self.max_file_size} bytes"
                }

            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                return {
                    "error": f"Unsupported file format. Supported formats: {', '.join(self.supported_extensions)}"
                }

            return {"valid": True, "file_size": file_size, "file_type": file_ext}

        except Exception as e:
            return {"error": f"Validation error: {str(e)}"}

    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from document"""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".pdf":
                return self._extract_pdf_text(file_path)
            elif file_ext == ".docx":
                return self._extract_docx_text(file_path)
            elif file_ext in [".txt", ".md"]:
                return self._extract_text_file(file_path)
            else:
                return {"error": f"Unsupported file format: {file_ext}"}

        except Exception as e:
            return {"error": f"Text extraction error: {str(e)}"}

    def _extract_pdf_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                return {"text": text, "pages": len(pdf_reader.pages)}

        except Exception as e:
            return {"error": f"PDF extraction error: {str(e)}"}

    def _extract_docx_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"

            return {"text": text, "paragraphs": len(doc.paragraphs)}

        except Exception as e:
            return {"error": f"DOCX extraction error: {str(e)}"}

    def _extract_text_file(self, file_path: str) -> Dict[str, Any]:
        """Extract text from plain text or markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Convert markdown to plain text if needed
            if file_path.endswith(".md"):
                content = markdown.markdown(content)
                # Remove HTML tags
                content = re.sub(r"<[^>]+>", "", content)

            return {"text": content, "lines": len(content.split("\n"))}

        except Exception as e:
            return {"error": f"Text file extraction error: {str(e)}"}

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces"""
        try:
            if not text.strip():
                return []

            # Clean and normalize text
            text = self._clean_text(text)

            # Split into sentences first
            sentences = self._split_into_sentences(text)

            chunks = []
            current_chunk = ""
            current_start = 0
            chunk_index = 0

            for i, sentence in enumerate(sentences):
                # Check if adding this sentence would exceed chunk size
                if (
                    len(current_chunk) + len(sentence) > self.chunk_size
                    and current_chunk
                ):
                    # Save current chunk
                    chunks.append(
                        {
                            "chunk_index": chunk_index,
                            "content": current_chunk.strip(),
                            "start_position": current_start,
                            "end_position": current_start + len(current_chunk),
                        }
                    )

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + sentence
                    current_start = (
                        current_start
                        + len(current_chunk)
                        - len(overlap_text)
                        - len(sentence)
                    )
                    chunk_index += 1

                    # Check max chunks limit
                    if chunk_index >= self.max_chunks:
                        break
                else:
                    current_chunk += sentence + " "

            # Add final chunk if there's content
            if current_chunk.strip() and chunk_index < self.max_chunks:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "content": current_chunk.strip(),
                        "start_position": current_start,
                        "end_position": current_start + len(current_chunk),
                    }
                )

            return chunks

        except Exception as e:
            return [{"error": f"Chunking error: {str(e)}"}]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", "", text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap :]

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document and return chunks with metadata"""
        try:
            # Validate document
            validation = self.validate_document(file_path)
            if "error" in validation:
                return validation

            # Extract text
            extraction = self.extract_text(file_path)
            if "error" in extraction:
                return extraction

            # Chunk text
            chunks = self.chunk_text(extraction["text"])
            if not chunks or "error" in chunks[0]:
                return {"error": "Failed to chunk document"}

            return {
                "chunks": chunks,
                "file_info": {
                    "file_size": validation["file_size"],
                    "file_type": validation["file_type"],
                    "total_chunks": len(chunks),
                },
            }

        except Exception as e:
            return {"error": f"Document processing error: {str(e)}"}

    def process_and_store_document(
        self, file_path: str, user_id: int, filename: str, db: Session
    ) -> Dict[str, Any]:
        """Process document and store with embeddings in database"""
        try:
            # Process document
            result = self.process_document(file_path)
            if "error" in result:
                return result

            # Create document record
            document = Document(
                user_id=user_id,
                filename=filename,
                file_path=file_path,
                file_size=result["file_info"]["file_size"],
                file_type=result["file_info"]["file_type"],
                status="processing",
            )
            db.add(document)
            db.flush()  # Get the document ID

            # Create chunks and embeddings
            chunks_created = 0
            for chunk_data in result["chunks"]:
                # Create chunk record
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    start_position=chunk_data["start_position"],
                    end_position=chunk_data["end_position"],
                )
                db.add(chunk)
                db.flush()  # Get the chunk ID

                # Create and store embedding
                embedding = self.embedding_service.create_embedding(
                    chunk_data["content"]
                )
                if embedding:
                    # Store embedding directly as list for PostgreSQL VECTOR type
                    chunk.embedding = embedding
                    chunks_created += 1

            # Update document status
            document.status = "processed"
            db.commit()

            return {
                "document_id": document.id,
                "chunks_created": chunks_created,
                "total_chunks": len(result["chunks"]),
                "file_info": result["file_info"],
                "status": "processed",
            }

        except Exception as e:
            db.rollback()
            return {"error": f"Error processing and storing document: {str(e)}"}

    def delete_document_chunks(self, document_id: int, db: Session) -> bool:
        """Delete all chunks and embeddings for a document"""
        try:
            # Delete chunks (embeddings will be deleted due to cascade)
            chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .all()
            )

            for chunk in chunks:
                db.delete(chunk)

            db.commit()
            return True

        except Exception as e:
            db.rollback()
            print(f"Error deleting document chunks: {e}")
            return False
