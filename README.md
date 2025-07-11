# DocuMind AI Assistant

An AI-powered document Q&A system that allows users to upload documents and ask natural language questions to get instant answers with source citations.

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Docker (optional, for PostgreSQL)

### Database Setup

#### Option 1: Using Docker (Recommended)

```bash
# Start PostgreSQL with Docker Compose
docker-compose up -d postgres

# The database will be available at:
# Host: localhost
# Port: 5432
# Database: documind
# Username: postgres
# Password: password
```

#### Option 2: Local PostgreSQL Installation

1. Install PostgreSQL on your system
2. Create a database:
   ```sql
   CREATE DATABASE documind;
   CREATE USER postgres WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE documind TO postgres;
   ```

### Application Setup

1. **Clone and install dependencies:**

   ```bash
   git clone <repository-url>
   cd docu-mind
   pip install -r requirements.txt
   ```

2. **Configure environment:**

   ```bash
   cp config.example.env .env
   # Edit .env with your settings
   ```

3. **Run database migrations:**

   ```bash
   alembic upgrade head
   ```

4. **Start the application:**
   ```bash
   python run.py
   ```

The application will be available at `http://localhost:8000`

### Optional: pgAdmin Setup

If you used Docker Compose, pgAdmin is also available at `http://localhost:5050`:

- Email: admin@documind.com
- Password: admin

---

## Test Suite

This repository also contains a comprehensive test suite for the DocuMind AI Assistant.

## Overview

The test suite covers all major components of the DocuMind system:

- **Document Upload & Processing**: Tests for file validation, chunking, and embedding creation
- **Question Answering**: Tests for semantic search, LLM integration, and answer generation
- **API Endpoints**: Tests for REST API functionality and error handling
- **Integration Workflows**: End-to-end tests for complete user workflows
- **Accuracy & Hallucination Detection**: Tests to ensure AI responses are grounded in source documents
- **Performance**: Tests for handling large documents and high query volumes

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_document_upload.py        # Document upload and processing tests
├── test_question_answering.py     # Question answering and semantic search tests
├── test_api_endpoints.py          # API endpoint tests
├── test_integration.py            # End-to-end integration tests
├── test_accuracy_and_hallucination.py  # Accuracy validation and hallucination detection
└── test_performance.py            # Performance and scalability tests
```

## Key Features Tested

### 1. Document Upload & Processing

- ✅ Multiple document format support (PDF, DOCX, TXT, MD)
- ✅ File size validation (50MB limit)
- ✅ Document chunking and text extraction
- ✅ User document isolation
- ✅ Error handling for invalid files

### 2. Question Answering

- ✅ Natural language question processing
- ✅ Semantic search with embeddings
- ✅ LLM integration for answer generation
- ✅ Source citations with similarity scores
- ✅ Confidence scoring
- ✅ Multi-document queries

### 3. API Endpoints

- ✅ Document upload endpoints
- ✅ Question asking endpoints
- ✅ Document library management
- ✅ User authentication and isolation
- ✅ Error handling and validation

### 4. Integration Workflows

- ✅ Complete upload → question → answer workflow
- ✅ Multi-user support
- ✅ Document processing accuracy
- ✅ Cross-document queries

### 5. Accuracy & Hallucination Detection

- ✅ Answer grounding validation
- ✅ Source consistency checking
- ✅ Fact verification
- ✅ Hallucination risk assessment
- ✅ Confidence adjustment based on validation

### 6. Performance & Scalability

- ✅ Large document processing
- ✅ Concurrent user support
- ✅ Response time optimization
- ✅ Memory usage monitoring
- ✅ Embedding caching
- ✅ Context optimization

## Running the Tests

### Prerequisites

```bash
# Install Python dependencies
pip install pytest pytest-cov pytest-mock

# Or install from requirements.txt
pip install -r requirements.txt
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=tests --cov-report=html

# Run specific test file
pytest tests/test_document_upload.py

# Run specific test class
pytest tests/test_document_upload.py::TestDocumentUpload

# Run specific test method
pytest tests/test_document_upload.py::TestDocumentUpload::test_upload_pdf_document
```

### Running Tests by Category

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only accuracy tests
pytest -m accuracy

# Run only hallucination detection tests
pytest -m hallucination

# Run fast tests only
pytest -m fast

# Run slow tests only
pytest -m slow
```

### Performance Testing

```bash
# Run performance tests with timing
pytest -m performance --durations=10

# Run with memory profiling (requires psutil)
pytest tests/test_performance.py::TestSystemScalability::test_memory_usage_with_large_documents
```

### Test Configuration

The test suite uses `pytest.ini` for configuration:

- **Test Discovery**: Automatically finds test files in the `tests/` directory
- **Markers**: Custom markers for categorizing tests
- **Output**: Verbose output with color coding
- **Warnings**: Suppresses deprecation warnings

## Test Data

The test suite includes comprehensive test data:

### Sample Documents

- Financial reports with revenue and employee data
- Employee satisfaction surveys
- Market analysis reports
- Legal contracts
- Large documents for performance testing

### Sample Questions

- Revenue and financial queries
- Employee and HR questions
- Market and competitive analysis
- Legal and contractual information
- Cross-document queries

### Expected Outcomes

- Keyword-based answer validation
- Source relevance checking
- Confidence score verification
- Error handling validation

## Mock Services

The test suite uses mock services to avoid external dependencies:

### EmbeddingService

- Mock embedding creation
- Mock semantic search
- Mock vector storage

### LLMService

- Mock answer generation
- Mock confidence scoring
- Mock reasoning generation

### DocumentProcessor

- Mock file validation
- Mock text extraction
- Mock chunking

## Test Fixtures

Shared fixtures in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_documents`: Pre-created test documents
- `mock_embedding_service`: Mock embedding service
- `mock_llm_service`: Mock LLM service
- `test_user_id`: Test user identifier
- `sample_questions`: Pre-defined test questions

## Performance Benchmarks

The test suite includes performance benchmarks:

### Document Processing

- **Small documents** (< 1MB): < 5 seconds
- **Medium documents** (1-10MB): < 30 seconds
- **Large documents** (10-50MB): < 60 seconds

### Question Answering

- **Single question**: < 5 seconds
- **Multiple questions**: < 25 seconds
- **Concurrent questions**: < 10 seconds

### System Scalability

- **Memory usage**: < 500MB increase for large documents
- **Concurrent users**: Support for 5+ simultaneous users
- **Response time**: Consistent < 3 seconds average

## Accuracy Validation

The test suite validates AI accuracy through:

### Grounding Validation

- Fact extraction from answers
- Source verification
- Consistency checking

### Hallucination Detection

- Source consistency analysis
- Unsourced claim detection
- Contradiction identification

### Confidence Scoring

- Risk level assessment
- Confidence adjustment
- Reliability indicators

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=tests --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_*.py` for files, `Test*` for classes, `test_*` for methods
2. **Use appropriate markers**: Mark tests as `unit`, `integration`, `performance`, etc.
3. **Add fixtures**: Use shared fixtures from `conftest.py` when possible
4. **Mock external dependencies**: Don't rely on external services
5. **Include assertions**: Every test should have meaningful assertions
6. **Document complex tests**: Add docstrings for complex test scenarios

## Test Coverage

The test suite aims for comprehensive coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Scalability and efficiency testing
- **Error Handling Tests**: Edge case and failure scenario testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Errors**: Check file permissions for temporary directories
3. **Memory Issues**: Reduce document sizes for performance tests
4. **Timeout Errors**: Increase timeout values for slow tests

### Debug Mode

```bash
# Run tests with debug output
pytest -v --tb=long

# Run specific test with debugger
pytest tests/test_document_upload.py::TestDocumentUpload::test_upload_pdf_document --pdb
```

## License

This test suite is part of the DocuMind AI Assistant project and follows the same license terms.
