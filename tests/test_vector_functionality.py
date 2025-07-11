"""
Tests for vector database functionality with PostgreSQL pgvector
"""

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, User, Document, DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.document_processor import DocumentProcessor
from app.services.qa_service import QuestionAnsweringService
from app.services.llm_service import LLMService
from app.auth import get_password_hash


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_vector.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create test database session"""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db_session):
    """Create test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("password123"),
    )
    db_session.add(user)
    db_session.commit()
    return user


class TestEmbeddingService:
    """Test embedding service with vector database"""

    def test_create_embedding(self):
        """Test creating embeddings"""
        service = EmbeddingService()

        # Test normal text
        embedding = service.create_embedding("This is a test sentence.")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

        # Test empty text
        empty_embedding = service.create_embedding("")
        assert len(empty_embedding) == 384
        assert all(x == 0.0 for x in empty_embedding)

    def test_batch_create_embeddings(self):
        """Test batch embedding creation"""
        service = EmbeddingService()
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = service.batch_create_embeddings(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_create_embedding(self, db_session):
        """Test creating embeddings"""
        service = EmbeddingService()

        # Create embedding
        embedding = service.create_embedding("Test content")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, (int, float)) for x in embedding)

        # Verify embedding was stored
        stored_embedding = service.get_embedding(db_session, chunk.id)
        assert stored_embedding is not None
        assert len(stored_embedding) == 384

    def test_search_similar(self, db_session, test_user):
        """Test vector similarity search"""
        service = EmbeddingService()

        # Create test document and chunks
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/test/path",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks with embeddings
        chunks_data = [
            "The company revenue is $10 million in 2024.",
            "There are 150 employees working at the company.",
            "The main office is located in New York.",
        ]

        for i, content in enumerate(chunks_data):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                content=content,
                start_position=i * 50,
                end_position=(i + 1) * 50,
            )
            db_session.add(chunk)
            db_session.flush()

            # Create and store embedding
            embedding = service.create_embedding(content)
            embedding_array = np.array(embedding, dtype=np.float32)
            chunk.embedding = embedding_array

        db_session.commit()

        # Test search
        query_embedding = service.create_embedding("What is the company revenue?")
        results = service.search_similar(
            db_session, query_embedding, user_id=test_user.id
        )

        assert len(results) > 0
        assert all("similarity" in result for result in results)
        assert all("content" in result for result in results)

    def test_get_search_stats(self, db_session, test_user):
        """Test getting search statistics"""
        service = EmbeddingService()

        # Create test data
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/test/path",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks with and without embeddings
        for i in range(3):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Content {i}",
                start_position=i * 50,
                end_position=(i + 1) * 50,
            )
            db_session.add(chunk)
            db_session.flush()

            # Add embedding to first two chunks only
            if i < 2:
                embedding = service.create_embedding(f"Content {i}")
                embedding_array = np.array(embedding, dtype=np.float32)
                chunk.embedding = embedding_array

        db_session.commit()

        # Get stats
        stats = service.get_search_stats(db_session, test_user.id)

        assert stats["total_chunks"] == 3
        assert stats["chunks_with_embeddings"] == 2
        assert stats["embedding_coverage"] == 2 / 3


class TestDocumentProcessor:
    """Test document processor with vector embeddings"""

    def test_process_and_store_document(self, db_session, test_user):
        """Test processing and storing document with embeddings"""
        processor = DocumentProcessor()

        # Create test file
        test_content = "This is a test document. It contains multiple sentences. We will process it into chunks."
        test_file_path = "/tmp/test_document.txt"

        with open(test_file_path, "w") as f:
            f.write(test_content)

        try:
            # Process and store document
            result = processor.process_and_store_document(
                test_file_path, test_user.id, "test_document.txt", db_session
            )

            assert "error" not in result
            assert result["document_id"] is not None
            assert result["chunks_created"] > 0
            assert result["status"] == "processed"

            # Verify document was created
            doc = (
                db_session.query(Document)
                .filter(Document.id == result["document_id"])
                .first()
            )
            assert doc is not None
            assert doc.user_id == test_user.id

            # Verify chunks were created with embeddings
            chunks = (
                db_session.query(DocumentChunk)
                .filter(DocumentChunk.document_id == doc.id)
                .all()
            )
            assert len(chunks) > 0

            # Check that embeddings were created
            chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
            assert len(chunks_with_embeddings) > 0

        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

    def test_delete_document_chunks(self, db_session, test_user):
        """Test deleting document chunks and embeddings"""
        processor = DocumentProcessor()

        # Create test document with chunks
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/test/path",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks with embeddings
        for i in range(3):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Content {i}",
                start_position=i * 50,
                end_position=(i + 1) * 50,
            )
            db_session.add(chunk)
            db_session.flush()

            # Add embedding
            embedding = np.array([0.1] * 384, dtype=np.float32)
            chunk.embedding = embedding

        db_session.commit()

        # Verify chunks exist
        chunks = (
            db_session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc.id)
            .all()
        )
        assert len(chunks) == 3

        # Delete chunks
        success = processor.delete_document_chunks(doc.id, db_session)
        assert success is True

        # Verify chunks were deleted
        chunks = (
            db_session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc.id)
            .all()
        )
        assert len(chunks) == 0


class TestQuestionAnsweringService:
    """Test question answering with vector search"""

    @patch("app.services.llm_service.LLMService.generate_answer")
    @patch("app.services.llm_service.LLMService.validate_answer_grounding")
    @patch("app.services.llm_service.LLMService.detect_hallucination")
    def test_ask_question_with_vector_search(
        self, mock_hallucination, mock_grounding, mock_generate, db_session, test_user
    ):
        """Test asking questions using vector search"""
        # Setup mocks
        mock_generate.return_value = {
            "answer": "The company revenue is $10 million in 2024.",
            "confidence": 0.9,
            "reasoning": "Based on the document content...",
            "model": "gpt-4",
        }
        mock_grounding.return_value = {"score": 0.85}
        mock_hallucination.return_value = {"hallucination_risk": "low"}

        # Create services
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        qa_service = QuestionAnsweringService(embedding_service, llm_service)

        # Create test document with chunks
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/test/path",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks with embeddings
        content = "The company revenue is $10 million in 2024."
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content=content,
            start_position=0,
            end_position=len(content),
        )
        db_session.add(chunk)
        db_session.flush()

        # Add embedding
        embedding = embedding_service.create_embedding(content)
        embedding_array = np.array(embedding, dtype=np.float32)
        chunk.embedding = embedding_array
        db_session.commit()

        # Ask question
        result = qa_service.ask_question(
            "What is the company revenue?", test_user.id, db_session
        )

        assert "error" not in result
        assert result["question"] == "What is the company revenue?"
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_search_similar_questions(self, db_session, test_user):
        """Test finding similar questions using vector similarity"""
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        qa_service = QuestionAnsweringService(embedding_service, llm_service)

        # Create test questions
        questions_data = [
            "What is the company revenue?",
            "How many employees work here?",
            "Where is the main office located?",
        ]

        for question_text in questions_data:
            question = Question(
                user_id=test_user.id,
                question_text=question_text,
                answer_text=f"Answer to {question_text}",
                confidence_score=0.8,
                processing_time=1.0,
            )
            db_session.add(question)

        db_session.commit()

        # Search for similar questions
        similar_questions = qa_service.search_similar_questions(
            "What is the revenue?", test_user.id, db_session
        )

        assert len(similar_questions) > 0
        assert all("similarity" in q for q in similar_questions)
        assert all("question_text" in q for q in similar_questions)


if __name__ == "__main__":
    pytest.main([__file__])
