import pytest
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock


# Shared test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing"""
    documents = {}

    # Financial report
    financial_file = os.path.join(temp_dir, "financial_report.txt")
    with open(financial_file, "w") as f:
        f.write(
            "The company reported revenue of $15 million in Q1 2024. Employee count increased to 150. The main office is in San Francisco."
        )
    documents["financial"] = financial_file

    # Employee report
    employee_file = os.path.join(temp_dir, "employee_report.txt")
    with open(employee_file, "w") as f:
        f.write(
            "Employee satisfaction scores reached 85%. The team size grew by 20%. Revenue per employee improved by 15%."
        )
    documents["employee"] = employee_file

    # Market analysis
    market_file = os.path.join(temp_dir, "market_analysis.txt")
    with open(market_file, "w") as f:
        f.write(
            "Market analysis shows 25% growth in the tech sector. Competitor analysis reveals three main players. Customer satisfaction is at 92%."
        )
    documents["market"] = market_file

    # Legal document
    legal_file = os.path.join(temp_dir, "legal_contract.txt")
    with open(legal_file, "w") as f:
        f.write(
            "This agreement is between Company A and Company B. The contract term is 24 months. The total value is $2.5 million."
        )
    documents["legal"] = legal_file

    return documents


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    service = Mock()
    service.create_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    service.embeddings = {}

    def store_embedding(doc_id, chunk_id, embedding):
        key = f"{doc_id}_{chunk_id}"
        service.embeddings[key] = embedding

    def search_similar(query_embedding, top_k=5):
        return [
            {
                "doc_id": 1,
                "chunk_id": 1,
                "similarity": 0.95,
                "content": "Relevant content from document 1",
            },
            {
                "doc_id": 1,
                "chunk_id": 2,
                "similarity": 0.87,
                "content": "More relevant content from document 1",
            },
            {
                "doc_id": 2,
                "chunk_id": 1,
                "similarity": 0.82,
                "content": "Relevant content from document 2",
            },
        ]

    service.store_embedding = store_embedding
    service.search_similar = search_similar
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service"""
    service = Mock()
    service.model = "gpt-4"

    def generate_answer(question, context_chunks, sources):
        return {
            "answer": "This is a comprehensive answer based on the provided context.",
            "confidence": 0.92,
            "sources": sources,
            "reasoning": "Based on the context provided, I can answer this question accurately.",
        }

    service.generate_answer = generate_answer
    return service


@pytest.fixture
def mock_document_processor():
    """Mock document processor"""
    processor = Mock()
    processor.supported_formats = [".pdf", ".docx", ".txt", ".md"]

    def validate_document(file_path):
        if not any(
            file_path.lower().endswith(fmt) for fmt in processor.supported_formats
        ):
            raise ValueError(
                f"Unsupported file format. Supported: {processor.supported_formats}"
            )

        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:
            raise ValueError("File too large. Maximum size: 50MB")

        return True

    def process_document(file_path):
        with open(file_path, "r") as f:
            content = f.read()
        sentences = content.split(". ")
        chunks = [s.strip() + "." for s in sentences if s.strip()]
        return chunks

    processor.validate_document = validate_document
    processor.process_document = process_document
    return processor


@pytest.fixture
def test_user_id():
    """Test user ID"""
    return "user123"


@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
        "What is the company revenue?",
        "How many employees are there?",
        "What is the employee satisfaction score?",
        "Where is the main office located?",
        "What are the key business metrics?",
        "What does the market analysis show?",
        "What are the contract terms?",
        "What is the customer satisfaction rate?",
    ]


@pytest.fixture
def expected_answers():
    """Expected answers for sample questions"""
    return {
        "What is the company revenue?": {
            "keywords": ["revenue", "million", "15", "$"],
            "expected_source": "financial",
        },
        "How many employees are there?": {
            "keywords": ["employee", "150", "count"],
            "expected_source": "financial",
        },
        "What is the employee satisfaction score?": {
            "keywords": ["satisfaction", "85", "%"],
            "expected_source": "employee",
        },
        "Where is the main office located?": {
            "keywords": ["san francisco", "office", "location"],
            "expected_source": "financial",
        },
        "What are the key business metrics?": {
            "keywords": ["revenue", "employee", "satisfaction"],
            "expected_source": "multiple",
        },
        "What does the market analysis show?": {
            "keywords": ["market", "growth", "25%", "tech"],
            "expected_source": "market",
        },
        "What are the contract terms?": {
            "keywords": ["contract", "24", "months", "agreement"],
            "expected_source": "legal",
        },
        "What is the customer satisfaction rate?": {
            "keywords": ["customer", "satisfaction", "92%"],
            "expected_source": "market",
        },
    }


# Test data fixtures
@pytest.fixture
def test_document_chunks():
    """Sample document chunks for testing"""
    return {
        1: [
            "The company reported revenue of $15 million in Q1 2024.",
            "Employee count increased to 150 employees.",
            "The main office is located in San Francisco, CA.",
        ],
        2: [
            "Employee satisfaction scores reached 85% this year.",
            "The team size grew by 20% compared to last year.",
            "Revenue per employee improved by 15%.",
        ],
        3: [
            "Market analysis shows 25% growth in the tech sector.",
            "Competitor analysis reveals three main players in the market.",
            "Customer satisfaction is at 92%.",
        ],
    }


@pytest.fixture
def mock_api_response():
    """Mock API response structure"""
    return {
        "success": True,
        "data": {},
        "message": "Operation completed successfully",
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def mock_error_response():
    """Mock error response structure"""
    return {
        "success": False,
        "error": "An error occurred",
        "message": "Operation failed",
        "timestamp": "2024-01-15T10:30:00Z",
    }


# Performance testing fixtures
@pytest.fixture
def large_document(temp_dir):
    """Create a large document for performance testing"""
    large_file = os.path.join(temp_dir, "large_document.txt")

    # Create a document with many sentences
    sentences = []
    for i in range(1000):
        sentences.append(
            f"This is sentence number {i} in the large document. It contains sample content for performance testing."
        )

    with open(large_file, "w") as f:
        f.write(" ".join(sentences))

    return large_file


@pytest.fixture
def multiple_documents(temp_dir):
    """Create multiple documents for batch testing"""
    documents = []

    for i in range(10):
        file_path = os.path.join(temp_dir, f"document_{i}.txt")
        with open(file_path, "w") as f:
            f.write(
                f"This is document {i}. It contains sample content for batch processing testing. Document {i} has specific information about topic {i}."
            )
        documents.append(file_path)

    return documents


# Validation fixtures
@pytest.fixture
def accuracy_test_cases():
    """Test cases for accuracy validation"""
    return [
        {
            "question": "What is the revenue?",
            "answer": "The revenue is $15 million.",
            "sources": ["The company reported revenue of $15 million in Q1 2024."],
            "expected_grounded": True,
            "expected_issues": [],
        },
        {
            "question": "What is the revenue?",
            "answer": "The revenue is $50 million.",
            "sources": ["The company reported revenue of $15 million in Q1 2024."],
            "expected_grounded": False,
            "expected_issues": ["Fact not found in sources: 50"],
        },
        {
            "question": "How many employees?",
            "answer": "There are 150 employees.",
            "sources": ["Employee count increased to 150 employees."],
            "expected_grounded": True,
            "expected_issues": [],
        },
    ]


@pytest.fixture
def hallucination_test_cases():
    """Test cases for hallucination detection"""
    return [
        {
            "answer": "The revenue is $15 million.",
            "sources": ["The company reported revenue of $15 million in Q1 2024."],
            "expected_hallucinated": False,
            "expected_confidence": 0.9,
        },
        {
            "answer": "The revenue is $50 million.",
            "sources": ["The company reported revenue of $15 million in Q1 2024."],
            "expected_hallucinated": True,
            "expected_confidence": 0.7,
        },
        {
            "answer": "The company has 500 employees.",
            "sources": ["Employee count increased to 150 employees."],
            "expected_hallucinated": True,
            "expected_confidence": 0.7,
        },
    ]


# Configuration fixtures
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    # Cleanup after test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "supported_formats": [".pdf", ".docx", ".txt", ".md"],
        "embedding_model": "text-embedding-ada-002",
        "llm_model": "gpt-4",
        "max_question_length": 1000,
        "top_k_results": 5,
        "min_confidence_threshold": 0.7,
        "max_documents_per_user": 100,
    }


# Utility functions for tests
def create_test_file(temp_dir, filename, content):
    """Helper function to create test files"""
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


def assert_response_structure(response, expected_fields=None):
    """Helper function to assert response structure"""
    if expected_fields is None:
        expected_fields = ["success", "message"]

    for field in expected_fields:
        assert field in response, f"Response missing field: {field}"


def assert_error_response(response, expected_status_code=None):
    """Helper function to assert error response"""
    assert "error" in response or response.get("success") == False
    if expected_status_code:
        assert response.get("status_code") == expected_status_code


def assert_success_response(response):
    """Helper function to assert success response"""
    assert response.get("success") == True or "error" not in response


# Custom markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line(
        "markers", "accuracy: mark test as an accuracy validation test"
    )
    config.addinivalue_line(
        "markers", "hallucination: mark test as a hallucination detection test"
    )
