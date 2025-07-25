"""
Cloudinary service for file uploads and management
"""

import os
import tempfile
import uuid
import time
import logging
from typing import Dict, Any, Optional
import cloudinary
import cloudinary.uploader
import cloudinary.api
from app.config import settings

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


class CloudinaryService:
    """Service for handling Cloudinary file uploads and management"""

    def __init__(self):
        self.cloud_name = settings.cloudinary_cloud_name
        self.api_key = settings.cloudinary_api_key
        self.api_secret = settings.cloudinary_api_secret
        self.folder = settings.cloudinary_folder

        # Log config (do not log secret)
        logging.info(
            f"Cloudinary config: cloud_name={self.cloud_name}, api_key={self.api_key}, folder={self.folder}"
        )

        # Configure Cloudinary
        if all([self.cloud_name, self.api_key, self.api_secret]):
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            self.is_configured = True
        else:
            self.is_configured = False

    def upload_file(
        self, file_content: bytes, filename: str, file_type: str
    ) -> Dict[str, Any]:
        """
        Upload a file to Cloudinary

        Args:
            file_content: File content as bytes
            filename: Original filename
            file_type: File type/extension

        Returns:
            Dict with upload result or error
        """
        try:
            logging.info(
                f"Uploading file: {filename}, size={len(file_content)} bytes, type={file_type}"
            )
            if not self.is_configured:
                logging.error(
                    "Cloudinary not configured. Please set cloudinary credentials."
                )
                return {
                    "error": "Cloudinary not configured. Please set cloudinary credentials."
                }

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # Generate unique filename with timestamp and UUID
                base_name = os.path.splitext(filename)[0]
                timestamp = int(time.time())
                unique_id = str(uuid.uuid4())[:8]
                unique_filename = f"{base_name}_{timestamp}_{unique_id}"
                logging.info(
                    f"Generated unique filename for Cloudinary: {unique_filename}"
                )

                # Upload to Cloudinary
                upload_result = cloudinary.uploader.upload(
                    temp_file_path,
                    folder=self.folder,
                    resource_type="auto",
                    public_id=unique_filename,
                    overwrite=False,
                )
                logging.info(f"Cloudinary upload result: {upload_result}")

                return {
                    "success": True,
                    "url": upload_result.get("secure_url"),
                    "public_id": upload_result.get("public_id"),
                    "file_size": upload_result.get("bytes"),
                    "format": upload_result.get("format"),
                    "resource_type": upload_result.get("resource_type"),
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except cloudinary.exceptions.Error as e:
            logging.error(f"Cloudinary exception: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                return {
                    "error": "Cloudinary authentication failed. Please check your API credentials."
                }
            elif "404" in str(e):
                return {"error": "Cloudinary resource not found."}
            else:
                return {"error": f"Cloudinary upload failed: {str(e)}"}
        except Exception as e:
            logging.error(f"Upload failed: {e}")
            return {"error": f"Upload failed: {str(e)}"}

    def delete_file(self, public_id: str) -> Dict[str, Any]:
        """
        Delete a file from Cloudinary

        Args:
            public_id: Cloudinary public ID of the file

        Returns:
            Dict with deletion result or error
        """
        try:
            if not self.is_configured:
                return {"error": "Cloudinary not configured"}

            result = cloudinary.uploader.destroy(public_id)

            if result.get("result") == "ok":
                return {"success": True, "message": "File deleted successfully"}
            else:
                return {"error": f"Deletion failed: {result.get('result')}"}

        except Exception as e:
            return {"error": f"Deletion failed: {str(e)}"}

    def get_file_info(self, public_id: str) -> Dict[str, Any]:
        """
        Get information about a file in Cloudinary

        Args:
            public_id: Cloudinary public ID of the file

        Returns:
            Dict with file information or error
        """
        try:
            if not self.is_configured:
                return {"error": "Cloudinary not configured"}

            result = cloudinary.api.resource(public_id)

            return {
                "success": True,
                "url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "file_size": result.get("bytes"),
                "format": result.get("format"),
                "created_at": result.get("created_at"),
            }

        except Exception as e:
            return {"error": f"Failed to get file info: {str(e)}"}

    def is_available(self) -> bool:
        """Check if Cloudinary is properly configured and available"""
        return self.is_configured and all(
            [self.cloud_name, self.api_key, self.api_secret]
        )
