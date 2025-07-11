#!/usr/bin/env python3
"""
Simple verification script for Cloudinary setup
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Mock heavy dependencies
sys.modules["sentence_transformers"] = type("MockModule", (), {})()
sys.modules["sklearn"] = type("MockModule", (), {})()
sys.modules["numpy"] = type("MockModule", (), {})()
sys.modules["torch"] = type("MockModule", (), {})()


# Mock specific classes
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        return [0.1] * 384  # Mock embedding


sys.modules["sentence_transformers"].SentenceTransformer = MockSentenceTransformer


def verify_cloudinary_setup():
    """Verify Cloudinary setup without heavy dependencies"""
    try:
        # Test imports
        print("Testing imports...")
        from app.services.cloudinary_service import CloudinaryService
        from app.config import settings

        print("✓ Cloudinary service imported successfully")

        # Test Cloudinary service initialization
        print("Testing Cloudinary service initialization...")
        service = CloudinaryService()
        print(f"✓ Cloudinary service initialized")
        print(f"  - Configured: {service.is_configured}")
        print(f"  - Available: {service.is_available()}")

        # Test configuration
        print("Testing configuration...")
        print(f"  - USE_CLOUDINARY: {getattr(settings, 'use_cloudinary', False)}")
        print(
            f"  - CLOUDINARY_CLOUD_NAME: {getattr(settings, 'cloudinary_cloud_name', 'Not set')}"
        )
        print(
            f"  - CLOUDINARY_API_KEY: {'Set' if getattr(settings, 'cloudinary_api_key', None) else 'Not set'}"
        )
        print(
            f"  - CLOUDINARY_API_SECRET: {'Set' if getattr(settings, 'cloudinary_api_secret', None) else 'Not set'}"
        )

        # Test document processor import (without heavy deps)
        print("Testing document processor...")
        from app.services.document_processor import DocumentProcessor

        print("✓ Document processor imported successfully")

        # Test models import
        print("Testing models...")
        from app.models import Document

        print("✓ Document model imported successfully")

        # Check if Cloudinary fields exist in model
        doc_fields = [attr for attr in dir(Document) if not attr.startswith("_")]
        cloudinary_fields = ["cloudinary_url", "cloudinary_public_id"]

        for field in cloudinary_fields:
            if field in doc_fields:
                print(f"✓ {field} field exists in Document model")
            else:
                print(f"✗ {field} field missing from Document model")

        print("\n=== Cloudinary Setup Verification Complete ===")
        print("✓ All core components are working")

        if service.is_available():
            print("✓ Cloudinary is properly configured and available")
            print("  You can now use Cloudinary for file uploads!")
        else:
            print("⚠ Cloudinary is not configured")
            print(
                "  Set USE_CLOUDINARY=true and configure credentials to enable Cloudinary"
            )
            print("  The system will fallback to local storage")

        return True

    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_cloudinary_setup()
    sys.exit(0 if success else 1)
