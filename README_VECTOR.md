# DocuMind AI Assistant - Vector Database with PostgreSQL pgvector

This document describes the vector database functionality implemented using PostgreSQL with the pgvector extension for semantic search and retrieval.

## Overview

The DocuMind AI Assistant now includes a comprehensive vector database system that provides:

- **Vector Database with Embeddings Storage**: PostgreSQL with pgvector extension
- **Semantic Search using Cosine Similarity**: Find relevant document chunks based on meaning
- **Top 3-5 Most Relevant Chunks per Query**: Configurable result ranking
- **Basic Metadata Tracking**: Document name, chunk index, timestamp
- **Response Generation**: Combine retrieved chunks with user queries using LLM
- **Source Citations**: Include document name and chunk references
- **Graceful Handling**: Handle "information not found" scenarios

## Architecture

### Database Schema

The system uses PostgreSQL with the pgvector extension to store vector embeddings directly in the database:

```sql
-- Document chunks with vector embeddings
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    embedding VECTOR(384),  -- 384-dimensional embeddings
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector index for fast similarity search
CREATE INDEX idx_chunk_embedding
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Key Components

1. **EmbeddingService**: Creates and manages vector embeddings
2. **DocumentProcessor**: Processes documents and creates embeddings
3. **QuestionAnsweringService**: Uses vector search for Q&A
4. **Vector Search API**: REST endpoints for semantic search

## Features

### 1. Basic Retrieval System

#### Vector Database with Embeddings Storage

- Uses PostgreSQL with pgvector extension
- Stores 384-dimensional embeddings (all-MiniLM-L6-v2 model)
- Automatic embedding creation during document processing
- Efficient storage with vector compression

#### Semantic Search using Cosine Similarity

- Converts user queries to embeddings
- Searches for similar document chunks using cosine similarity
- Configurable similarity threshold (default: 0.7)
- Returns top-k most relevant results (default: 5)

#### Return Top 3-5 Most Relevant Chunks per Query

- Configurable via `top_k_results` setting
- Results ranked by similarity score
- Includes chunk content and metadata
- User-specific filtering

#### Basic Metadata Tracking

- Document name and ID
- Chunk index and position
- Creation timestamp
- Similarity scores
- User ownership

### 2. Response Generation

#### Combine Retrieved Chunks with User Query

- Retrieves relevant chunks using vector search
- Combines chunks with original query
- Sends to LLM for answer generation
- Maintains context and relevance

#### Generate Coherent Answers using LLM

- Uses OpenAI GPT-4 or configured model
- Generates natural language responses
- Maintains conversation flow
- Handles complex queries

#### Include Source Citations

- Document name and chunk reference
- Similarity scores for transparency
- Chunk content preview
- Confidence metrics

#### Handle "Information Not Found" Scenarios

- Graceful handling when no relevant chunks found
- Suggests rephrasing or uploading more documents
- Returns helpful error messages
- Maintains user experience

## API Endpoints

### Vector Search

```http
POST /api/v1/search
Content-Type: application/x-www-form-urlencoded

query=What is the company revenue?&top_k=5
```

Response:

```json
{
  "query": "What is the company revenue?",
  "results": [
    {
      "chunk_id": 1,
      "document_id": 1,
      "content": "The company revenue is $10 million in 2024.",
      "filename": "financial_report.pdf",
      "similarity_score": 0.95,
      "chunk_index": 0
    }
  ],
  "total_results": 1,
  "search_metadata": {
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "top_k_requested": 5
  }
}
```

### Similar Questions

```http
POST /api/v1/questions/similar
Content-Type: application/x-www-form-urlencoded

question=What is the revenue?&limit=5
```

### Enhanced Document Upload

```http
POST /api/v1/upload
Content-Type: multipart/form-data

file=@document.pdf
```

Response includes embedding creation status:

```json
{
  "success": true,
  "document_id": 1,
  "filename": "document.pdf",
  "chunks_created": 15,
  "total_chunks": 15,
  "file_size": 1024000,
  "status": "processed"
}
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/documind
USE_POSTGRESQL=true

# Vector Search
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
TOP_K_RESULTS=5
VECTOR_INDEX_LISTS=100

# AI Services
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4
```

### Settings

```python
# Vector Search Configuration
embedding_model: str = "all-MiniLM-L6-v2"
embedding_dimension: int = 384
similarity_threshold: float = 0.7
top_k_results: int = 5
vector_index_lists: int = 100
```

## Setup Instructions

### 1. Install PostgreSQL with pgvector

#### Ubuntu/Debian

```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql-14 postgresql-14-pgvector

# Or for PostgreSQL 15
sudo apt-get install postgresql-15 postgresql-15-pgvector
```

#### macOS

```bash
# Install with Homebrew
brew install postgresql pgvector
```

#### Docker

```bash
# Use pgvector image
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_DB=documind \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg15
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Setup Database

```bash
# Run database setup
python setup_database.py
```

### 4. Run Migrations

```bash
# Initialize Alembic
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial migration with pgvector"

# Apply migration
alembic upgrade head
```

## Usage Examples

### 1. Upload and Process Documents

```python
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

# Initialize services
embedding_service = EmbeddingService()
processor = DocumentProcessor(embedding_service)

# Process document with embeddings
result = processor.process_and_store_document(
    file_path="/path/to/document.pdf",
    user_id=1,
    filename="document.pdf",
    db=db_session
)
```

### 2. Semantic Search

```python
from app.services.embedding_service import EmbeddingService

# Initialize service
service = EmbeddingService()

# Create query embedding
query_embedding = service.create_embedding("What is the company revenue?")

# Search for similar chunks
results = service.search_similar(
    db=db_session,
    query_embedding=query_embedding,
    user_id=1,
    top_k=5
)
```

### 3. Question Answering

```python
from app.services.qa_service import QuestionAnsweringService

# Initialize service
qa_service = QuestionAnsweringService(embedding_service, llm_service)

# Ask question with vector search
result = await qa_service.ask_question(
    question="What is the company revenue?",
    user_id=1,
    db=db_session
)
```

## Testing

### Run Vector Functionality Tests

```bash
# Run specific vector tests
pytest tests/test_vector_functionality.py -v

# Run all tests
pytest tests/ -v
```

### Test Coverage

The vector functionality includes tests for:

- Embedding creation and storage
- Vector similarity search
- Document processing with embeddings
- Question answering with vector search
- API endpoints for vector search
- Error handling and edge cases

## Performance Considerations

### Vector Index Optimization

- Uses IVFFlat index for fast similarity search
- Configurable number of lists (default: 100)
- Optimized for cosine similarity operations
- Automatic index maintenance

### Memory Management

- Embeddings stored in database, not memory
- Efficient batch processing
- Configurable chunk sizes
- Automatic cleanup of old embeddings

### Scalability

- Horizontal scaling with PostgreSQL clustering
- Vector operations distributed across nodes
- Configurable similarity thresholds
- Efficient query optimization

## Troubleshooting

### Common Issues

1. **pgvector extension not found**

   ```bash
   # Check if extension is available
   psql -d documind -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

   # Install if missing
   sudo apt-get install postgresql-14-pgvector
   ```

2. **Vector dimension mismatch**

   - Ensure `embedding_dimension` matches model output
   - Default is 384 for all-MiniLM-L6-v2
   - Check model configuration

3. **Low similarity scores**

   - Adjust `similarity_threshold` in settings
   - Check embedding model quality
   - Verify document preprocessing

4. **Slow search performance**
   - Optimize vector index parameters
   - Increase `vector_index_lists` for larger datasets
   - Monitor query performance

### Debug Mode

Enable debug logging:

```python
# In config.py
debug: bool = True
```

Check logs for:

- Embedding creation status
- Vector search performance
- Database connection issues
- LLM response times

## Future Enhancements

### Planned Features

1. **Advanced Vector Operations**

   - Support for multiple embedding models
   - Dynamic similarity thresholds
   - Hybrid search (vector + keyword)

2. **Performance Optimizations**

   - Vector caching layer
   - Batch similarity computations
   - Distributed vector search

3. **Enhanced Metadata**

   - Semantic chunk classification
   - Topic modeling integration
   - Temporal relevance scoring

4. **Advanced Search**
   - Multi-modal embeddings (text + images)
   - Contextual similarity
   - Personalized search ranking

## Contributing

When contributing to the vector functionality:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Consider performance implications
5. Test with different embedding models

## License

This vector database functionality is part of the DocuMind AI Assistant project and follows the same licensing terms.
