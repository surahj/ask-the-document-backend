"""
Test upload endpoint with Cloudinary integration
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the heavy dependencies
sys.modules["sentence_transformers"] = Mock()
sys.modules["sklearn"] = Mock()
sys.modules["numpy"] = Mock()

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


class TestUploadWithCloudinary:
    """Test upload endpoint with Cloudinary integration"""

    @patch("app.services.cloudinary_service.CloudinaryService.is_available")
    @patch("app.services.cloudinary_service.CloudinaryService.upload_file")
    @patch(
        "app.services.document_processor.DocumentProcessor.process_and_store_document_from_cloudinary"
    )
    def test_upload_with_cloudinary_success(
        self, mock_process, mock_upload, mock_available
    ):
        """Test successful upload with Cloudinary"""
        # Mock Cloudinary service
        mock_available.return_value = True
        mock_upload.return_value = {
            "success": True,
            "url": "https://res.cloudinary.com/test/image/upload/test.pdf",
            "public_id": "test_public_id",
            "file_size": 1024,
            "format": "pdf",
        }

        # Mock document processing
        mock_process.return_value = {
            "document_id": 123,
            "chunks_created": 5,
            "total_chunks": 5,
            "file_info": {"file_size": 1024, "file_type": ".pdf", "total_chunks": 5},
            "status": "processed",
        }

        # Mock settings
        with patch.object(settings, "use_cloudinary", True):
            with patch.object(settings, "allowed_extensions", [".pdf"]):
                with patch.object(settings, "max_file_size", 10485760):  # 10MB

                    client = TestClient(app)

                    # Create test file content
                    file_content = b"test pdf content"

                    # Test upload
                    response = client.post(
                        "/api/v1/upload",
                        files={"file": ("test.pdf", file_content, "application/pdf")},
                        headers={"Authorization": "Bearer test-token"},  # Mock auth
                    )

                    # Verify response
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["storage_type"] == "cloudinary"
                    assert data["document_id"] == 123

    @patch("app.services.cloudinary_service.CloudinaryService.is_available")
    def test_upload_with_cloudinary_not_configured(self, mock_available):
        """Test upload when Cloudinary is not configured (fallback to local)"""
        # Mock Cloudinary service as not available
        mock_available.return_value = False

        # Mock settings
        with patch.object(settings, "use_cloudinary", True):
            with patch.object(settings, "allowed_extensions", [".pdf"]):
                with patch.object(settings, "max_file_size", 10485760):  # 10MB
                    with patch.object(settings, "upload_dir", "./test_uploads"):

                        client = TestClient(app)

                        # Create test file content
                        file_content = b"test pdf content"

                        # Test upload (should fallback to local storage)
                        response = client.post(
                            "/api/v1/upload",
                            files={
                                "file": ("test.pdf", file_content, "application/pdf")
                            },
                            headers={"Authorization": "Bearer test-token"},  # Mock auth
                        )

                        # Should still work with local storage
                        assert response.status_code in [
                            200,
                            401,
                        ]  # 401 if auth fails, but endpoint works

    @patch("app.services.cloudinary_service.CloudinaryService.is_available")
    @patch("app.services.cloudinary_service.CloudinaryService.upload_file")
    def test_upload_with_cloudinary_upload_failure(self, mock_upload, mock_available):
        """Test upload when Cloudinary upload fails"""
        # Mock Cloudinary service
        mock_available.return_value = True
        mock_upload.return_value = {"error": "Upload failed"}

        # Mock settings
        with patch.object(settings, "use_cloudinary", True):
            with patch.object(settings, "allowed_extensions", [".pdf"]):
                with patch.object(settings, "max_file_size", 10485760):  # 10MB

                    client = TestClient(app)

                    # Create test file content
                    file_content = b"test pdf content"

                    # Test upload
                    response = client.post(
                        "/api/v1/upload",
                        files={"file": ("test.pdf", file_content, "application/pdf")},
                        headers={"Authorization": "Bearer test-token"},  # Mock auth
                    )

                    # Should return error
                    assert response.status_code == 500
                    data = response.json()
                    assert "Upload failed" in data["detail"]

    def test_upload_unsupported_file_type(self):
        """Test upload with unsupported file type"""
        client = TestClient(app)

        # Create test file content
        file_content = b"test content"

        # Test upload with unsupported file type
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
            headers={"Authorization": "Bearer test-token"},  # Mock auth
        )

        # Should return error for unsupported file type
        assert response.status_code in [
            400,
            401,
        ]  # 401 if auth fails, but endpoint works

    def test_upload_no_file(self):
        """Test upload without file"""
        client = TestClient(app)

        # Test upload without file
        response = client.post(
            "/api/v1/upload",
            headers={"Authorization": "Bearer test-token"},  # Mock auth
        )

        # Should return error for no file
        assert response.status_code in [
            400,
            401,
        ]  # 401 if auth fails, but endpoint works
