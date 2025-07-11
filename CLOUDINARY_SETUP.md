# Cloudinary Integration Setup

This document explains how to set up and use Cloudinary for file uploads in DocuMind AI Assistant.

## Overview

The application now supports both local file storage and Cloudinary cloud storage for document uploads. When Cloudinary is configured, files are uploaded to Cloudinary instead of being stored locally, and the Cloudinary URLs are saved in the database.

## Benefits of Using Cloudinary

1. **Scalability**: No local storage limitations
2. **Reliability**: Cloud-based storage with high availability
3. **Performance**: Fast global CDN delivery
4. **Cost-effective**: Pay only for what you use
5. **Security**: Secure file access with signed URLs

## Setup Instructions

### 1. Create a Cloudinary Account

1. Go to [Cloudinary](https://cloudinary.com/) and sign up for a free account
2. After signing up, you'll get your Cloud Name, API Key, and API Secret from your dashboard

### 2. Configure Environment Variables

Copy the example environment file and update it with your Cloudinary credentials:

```bash
cp config.example.env .env
```

Edit the `.env` file and add your Cloudinary credentials:

```env
# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME="your-cloudinary-cloud-name"
CLOUDINARY_API_KEY="your-cloudinary-api-key"
CLOUDINARY_API_SECRET="your-cloudinary-api-secret"
CLOUDINARY_FOLDER="documind"
USE_CLOUDINARY=true  # Set to true to use Cloudinary instead of local storage
```

### 3. Run Database Migration

Run the migration script to add Cloudinary fields to the database:

```bash
python3 migrate_schema.py
```

### 4. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you're running out of disk space and can't install heavy ML dependencies (sentence-transformers, scikit-learn, numpy), the application will automatically fall back to mock embeddings for testing purposes. The Cloudinary integration will still work perfectly.

## Usage

### Automatic Storage Selection

The application automatically chooses between local storage and Cloudinary based on the `USE_CLOUDINARY` setting:

- If `USE_CLOUDINARY=true` and Cloudinary credentials are configured: Files are uploaded to Cloudinary
- Otherwise: Files are stored locally (fallback behavior)

### API Response

When uploading files, the API response includes a `storage_type` field indicating where the file was stored:

```json
{
  "success": true,
  "document_id": 123,
  "filename": "document.pdf",
  "chunks_created": 5,
  "total_chunks": 5,
  "file_size": 1024000,
  "status": "processed",
  "storage_type": "cloudinary" // or "local"
}
```

## Database Schema Changes

The `documents` table now includes these additional fields:

- `cloudinary_url`: The secure URL of the file in Cloudinary
- `cloudinary_public_id`: The public ID used to manage the file in Cloudinary
- `file_path`: Made nullable to support Cloudinary-only storage

## File Processing

When using Cloudinary:

1. Files are uploaded to Cloudinary first
2. The file is downloaded temporarily for text extraction and processing
3. The Cloudinary URL and public ID are stored in the database
4. Temporary files are cleaned up automatically

## File Deletion

When deleting documents:

- If the document has a `cloudinary_public_id`, the file is deleted from Cloudinary
- If the document has a local `file_path`, the local file is deleted
- Database records are always cleaned up regardless of storage type

## Testing

Run the Cloudinary integration tests:

```bash
python3 -m pytest tests/test_cloudinary_simple.py -v
```

### Verification Script

Use the verification script to check your setup:

```bash
python3 verify_cloudinary_setup.py
```

This will test all components and tell you if Cloudinary is properly configured.

## Configuration Options

| Environment Variable    | Description                | Default    |
| ----------------------- | -------------------------- | ---------- |
| `CLOUDINARY_CLOUD_NAME` | Your Cloudinary cloud name | None       |
| `CLOUDINARY_API_KEY`    | Your Cloudinary API key    | None       |
| `CLOUDINARY_API_SECRET` | Your Cloudinary API secret | None       |
| `CLOUDINARY_FOLDER`     | Folder name in Cloudinary  | "documind" |
| `USE_CLOUDINARY`        | Enable Cloudinary storage  | false      |

## Lightweight Mode

If you're running out of disk space or don't need full ML capabilities:

1. **The application will work without heavy ML dependencies** - it will use mock embeddings
2. **Cloudinary integration will work perfectly** - file uploads and storage will function normally
3. **Text processing will work** - document parsing and chunking will function
4. **Vector search will use mock similarities** - for testing purposes

To run in lightweight mode, simply don't install the heavy dependencies:

```bash
# Install only the essential dependencies
pip install fastapi uvicorn python-multipart sqlalchemy psycopg2-binary cloudinary requests python-dotenv
```

## Troubleshooting

### Common Issues

1. **"Cloudinary not configured" error**

   - Check that all Cloudinary credentials are set in your `.env` file
   - Verify that `USE_CLOUDINARY=true`

2. **Upload failures**

   - Check your Cloudinary account limits
   - Verify your API credentials are correct
   - Check network connectivity

3. **File processing errors**

   - Ensure the file format is supported
   - Check file size limits
   - Verify Cloudinary account has sufficient storage

4. **"Heavy ML dependencies not available" warning**
   - This is normal in lightweight mode
   - The application will still work with mock embeddings
   - Install full dependencies if you need real ML capabilities

### Debug Mode

Enable debug mode to see detailed Cloudinary logs:

```env
DEBUG=true
```

## Security Considerations

1. **API Credentials**: Keep your Cloudinary API credentials secure
2. **File Access**: Cloudinary URLs are secure by default
3. **Cleanup**: Files are automatically deleted when documents are removed
4. **Validation**: File types and sizes are validated before upload

## Cost Considerations

- Cloudinary offers a generous free tier
- Monitor your usage in the Cloudinary dashboard
- Consider setting up usage alerts
- Files are automatically deleted when documents are removed to save storage costs

## Migration from Local Storage

If you have existing documents stored locally and want to migrate to Cloudinary:

1. Set up Cloudinary as described above
2. The system will continue to work with existing local files
3. New uploads will use Cloudinary
4. Existing local files will remain accessible

## Support

For Cloudinary-specific issues:

- Check the [Cloudinary documentation](https://cloudinary.com/documentation)
- Review your Cloudinary dashboard for usage and limits
- Contact Cloudinary support if needed

For application-specific issues:

- Check the application logs
- Review the test suite for examples
- Ensure all dependencies are installed correctly
- Run the verification script to diagnose issues
