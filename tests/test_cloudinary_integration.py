"""
Test Cloudinary integration
"""

import pytest
from unittest.mock import Mock, patch
from app.services.cloudinary_service import CloudinaryService
from app.config import settings


class TestCloudinaryService:
    """Test Cloudinary service functionality"""

    def test_cloudinary_not_configured(self):
        """Test when Cloudinary is not configured"""
        with patch.object(settings, "cloudinary_cloud_name", None):
            with patch.object(settings, "cloudinary_api_key", None):
                with patch.object(settings, "cloudinary_api_secret", None):
                    service = CloudinaryService()
                    assert not service.is_available()
                    assert not service.is_configured

    def test_cloudinary_configured(self):
        """Test when Cloudinary is properly configured"""
        with patch.object(settings, "cloudinary_cloud_name", "test-cloud"):
            with patch.object(settings, "cloudinary_api_key", "test-key"):
                with patch.object(settings, "cloudinary_api_secret", "test-secret"):
                    service = CloudinaryService()
                    assert service.is_available()
                    assert service.is_configured

    @patch("cloudinary.uploader.upload")
    def test_upload_file_success(self, mock_upload):
        """Test successful file upload to Cloudinary"""
        mock_upload.return_value = {
            "secure_url": "https://res.cloudinary.com/test/image/upload/test.jpg",
            "public_id": "test_public_id",
            "bytes": 1024,
            "format": "pdf",
            "resource_type": "image",
        }

        with patch.object(settings, "cloudinary_cloud_name", "test-cloud"):
            with patch.object(settings, "cloudinary_api_key", "test-key"):
                with patch.object(settings, "cloudinary_api_secret", "test-secret"):
                    service = CloudinaryService()

                    result = service.upload_file(
                        b"test file content", "test.pdf", ".pdf"
                    )

                    assert result["success"] is True
                    assert "url" in result
                    assert "public_id" in result
                    assert result["file_size"] == 1024

    @patch("cloudinary.uploader.upload")
    def test_upload_file_failure(self, mock_upload):
        """Test file upload failure"""
        mock_upload.side_effect = Exception("Upload failed")

        with patch.object(settings, "cloudinary_cloud_name", "test-cloud"):
            with patch.object(settings, "cloudinary_api_key", "test-key"):
                with patch.object(settings, "cloudinary_api_secret", "test-secret"):
                    service = CloudinaryService()

                    result = service.upload_file(
                        b"test file content", "test.pdf", ".pdf"
                    )

                    assert "error" in result
                    assert "Upload failed" in result["error"]

    @patch("cloudinary.uploader.destroy")
    def test_delete_file_success(self, mock_destroy):
        """Test successful file deletion from Cloudinary"""
        mock_destroy.return_value = {"result": "ok"}

        with patch.object(settings, "cloudinary_cloud_name", "test-cloud"):
            with patch.object(settings, "cloudinary_api_key", "test-key"):
                with patch.object(settings, "cloudinary_api_secret", "test-secret"):
                    service = CloudinaryService()

                    result = service.delete_file("test_public_id")

                    assert result["success"] is True
                    assert "message" in result

    @patch("cloudinary.uploader.destroy")
    def test_delete_file_failure(self, mock_destroy):
        """Test file deletion failure"""
        mock_destroy.return_value = {"result": "not found"}

        with patch.object(settings, "cloudinary_cloud_name", "test-cloud"):
            with patch.object(settings, "cloudinary_api_key", "test-key"):
                with patch.object(settings, "cloudinary_api_secret", "test-secret"):
                    service = CloudinaryService()

                    result = service.delete_file("test_public_id")

                    assert "error" in result
                    assert "not found" in result["error"]
