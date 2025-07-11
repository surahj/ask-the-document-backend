"""
Simplified Frontend API Integration Tests
Tests core backend API endpoints that the frontend expects to work with
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import get_db, Base
from app.models import User, Document, DocumentChunk, Question
from app.auth import create_access_token, get_password_hash


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_frontend_simple.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("password123"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers"""
    token = create_access_token(data={"sub": test_user.email})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestAuthenticationEndpoints:
    """Test authentication endpoints that frontend uses"""

    def test_register_user(self, client):
        """Test user registration endpoint"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "password123",
                "name": "New User",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "user" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["name"] == "New User"

    def test_login_user(self, client, test_user):
        """Test user login endpoint"""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "wrong@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_get_user_profile(self, client, auth_headers):
        """Test getting user profile"""
        response = client.get("/api/v1/auth/profile", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "user" in data
        assert data["user"]["email"] == "test@example.com"

    def test_logout_user(self, client, auth_headers):
        """Test user logout endpoint"""
        response = client.post("/api/v1/auth/logout", headers=auth_headers)
        assert response.status_code == 200


class TestDocumentUploadEndpoints:
    """Test document upload endpoints that frontend uses"""

    def test_upload_txt_document(self, client, auth_headers):
        """Test uploading a TXT document"""
        txt_content = b"This is a sample text document for testing."

        with patch(
            "app.services.document_processor.DocumentProcessor.process_document"
        ) as mock_process:
            mock_process.return_value = {
                "chunks": [
                    {
                        "chunk_index": 0,
                        "content": "This is a sample text document for testing.",
                        "start_position": 0,
                        "end_position": 42,
                    }
                ],
                "file_info": {"file_size": len(txt_content)},
            }

            with patch(
                "app.services.embedding_service.EmbeddingService.create_embedding"
            ) as mock_embedding:
                mock_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

                files = {"file": ("test.txt", BytesIO(txt_content), "text/plain")}
                response = client.post(
                    "/api/v1/documents/upload", files=files, headers=auth_headers
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_upload_unsupported_format(self, client, auth_headers):
        """Test uploading an unsupported file format"""
        files = {
            "file": ("test.exe", BytesIO(b"binary content"), "application/octet-stream")
        }
        response = client.post(
            "/api/v1/documents/upload", files=files, headers=auth_headers
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_upload_no_file(self, client, auth_headers):
        """Test uploading without a file"""
        response = client.post("/api/v1/documents/upload", headers=auth_headers)
        assert response.status_code == 422  # Validation error


class TestDocumentManagementEndpoints:
    """Test document management endpoints that frontend uses"""

    def test_get_user_documents(self, client, auth_headers, test_user, db_session):
        """Test getting user documents with pagination"""
        # Create test documents
        doc1 = Document(
            user_id=test_user.id,
            filename="test1.pdf",
            file_path="/tmp/test1.pdf",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        doc2 = Document(
            user_id=test_user.id,
            filename="test2.docx",
            file_path="/tmp/test2.docx",
            file_size=2048,
            file_type=".docx",
            status="processed",
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        response = client.get(
            "/api/v1/documents?page=1&page_size=10", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "pagination" in data
        assert len(data["documents"]) == 2
        assert data["pagination"]["total_count"] == 2

    def test_delete_document(self, client, auth_headers, test_user, db_session):
        """Test deleting a document"""
        # Create a test document
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.commit()

        response = client.delete(f"/api/v1/documents/{doc.id}", headers=auth_headers)
        assert response.status_code == 200

        # Verify document is deleted
        deleted_doc = db_session.query(Document).filter_by(id=doc.id).first()
        assert deleted_doc is None

    def test_delete_nonexistent_document(self, client, auth_headers):
        """Test deleting a document that doesn't exist"""
        response = client.delete("/api/v1/documents/999", headers=auth_headers)
        assert response.status_code == 404


class TestChatEndpoints:
    """Test chat/question answering endpoints that frontend uses"""

    def test_ask_question_success(self, client, auth_headers, test_user, db_session):
        """Test asking a question successfully"""
        # Create a test document with chunks
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.commit()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="The company revenue is $10 million in 2024.",
            start_position=0,
            end_position=45,
        )
        db_session.add(chunk)
        db_session.commit()

        with patch(
            "app.services.qa_service.QuestionAnsweringService.ask_question"
        ) as mock_qa:
            mock_qa.return_value = {
                "question": "What is the company revenue?",
                "answer": "The company revenue is $10 million in 2024.",
                "confidence": 0.95,
                "sources": [
                    {
                        "doc_id": doc.id,
                        "chunk_id": chunk.id,
                        "similarity": 0.95,
                        "content": "The company revenue is $10 million in 2024.",
                        "filename": "test.pdf",
                    }
                ],
                "reasoning": "Based on the document content...",
                "user_id": test_user.id,
            }

            response = client.post(
                "/api/v1/chat/ask",
                data={"question": "What is the company revenue?"},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "confidence" in data
            assert "sources" in data

    def test_ask_empty_question(self, client, auth_headers):
        """Test asking an empty question"""
        response = client.post(
            "/api/v1/chat/ask", data={"question": ""}, headers=auth_headers
        )
        assert response.status_code == 400

    def test_get_chat_history(self, client, auth_headers, test_user, db_session):
        """Test getting chat history"""
        # Create test questions
        questions = [
            Question(
                user_id=test_user.id,
                question="What is the revenue?",
                answer="The revenue is $10 million.",
                confidence=0.95,
            ),
            Question(
                user_id=test_user.id,
                question="How many employees?",
                answer="There are 150 employees.",
                confidence=0.88,
            ),
        ]
        db_session.add_all(questions)
        db_session.commit()

        response = client.get("/api/v1/chat/history?limit=10", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) == 2


class TestHealthAndStatsEndpoints:
    """Test health check and system stats endpoints"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_system_stats(self, client, auth_headers, test_user, db_session):
        """Test system statistics endpoint"""
        # Create some test data
        doc = Document(
            user_id=test_user.id,
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            file_type=".pdf",
            status="processed",
        )
        db_session.add(doc)
        db_session.commit()

        response = client.get("/api/v1/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "total_questions" in data


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_unauthorized_access(self, client):
        """Test accessing protected endpoints without authentication"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 401

    def test_invalid_token(self, client):
        """Test accessing with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/documents", headers=headers)
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
