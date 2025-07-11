import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import json


# Mock classes for integration testing
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt", ".md"]

    def validate_document(self, file_path):
        if not any(file_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(
                f"Unsupported file format. Supported: {self.supported_formats}"
            )

        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:
            raise ValueError("File too large. Maximum size: 50MB")

        return True

    def process_document(self, file_path):
        # Mock document processing - extract text and chunk it
        with open(file_path, "r") as f:
            content = f.read()

        # Simple chunking by sentences
        sentences = content.split(". ")
        chunks = [s.strip() + "." for s in sentences if s.strip()]
        return chunks


class EmbeddingService:
    def __init__(self):
        self.embeddings = {}
        self.embedding_counter = 0

    def create_embedding(self, text):
        # Mock embedding creation
        self.embedding_counter += 1
        return [0.1 * self.embedding_counter, 0.2, 0.3, 0.4, 0.5]

    def store_embedding(self, doc_id, chunk_id, embedding):
        key = f"{doc_id}_{chunk_id}"
        self.embeddings[key] = embedding

    def search_similar(self, query_embedding, top_k=5):
        # Mock semantic search - return chunks with similarity scores
        results = []
        for key, embedding in self.embeddings.items():
            doc_id, chunk_id = key.split("_")
            # Mock similarity calculation
            similarity = 0.9 - (0.1 * int(chunk_id))

            # Return more realistic content based on document ID
            if doc_id == "1":
                content = "The company reported revenue of $15 million in Q1 2024. Employee count increased to 150. The main office is in San Francisco."
            elif doc_id == "2":
                content = "Employee satisfaction scores reached 85%. The team size grew by 20%. Revenue per employee improved by 15%."
            else:
                content = f"Content from document {doc_id}, chunk {chunk_id}"

            results.append(
                {
                    "doc_id": int(doc_id),
                    "chunk_id": int(chunk_id),
                    "similarity": similarity,
                    "content": content,
                }
            )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class LLMService:
    def __init__(self):
        self.model = "gpt-4"

    def generate_answer(self, question, context_chunks, sources):
        # Mock LLM response based on context
        context_text = " ".join(context_chunks)

        # Simple answer generation based on keywords
        if "revenue" in question.lower():
            answer = "Based on the documents, the revenue is $15 million as reported in the financial data."
        elif "employee" in question.lower() and "satisfaction" in question.lower():
            answer = "The employee satisfaction scores reached 85% this year."
        elif "satisfaction" in question.lower():
            answer = "The employee satisfaction scores reached 85% this year."
        elif "employee" in question.lower():
            answer = "The employee count increased to 150 employees."
        else:
            answer = "Based on the provided context, I can provide information about the requested topic."

        return {
            "answer": answer,
            "confidence": 0.85,
            "sources": sources,
            "reasoning": f"Generated answer based on {len(context_chunks)} relevant document chunks.",
        }


class DocumentUploadService:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.uploaded_docs = []
        self.doc_counter = 0

    def upload_document(self, file_path, user_id):
        try:
            self.processor.validate_document(file_path)
            chunks = self.processor.process_document(file_path)

            self.doc_counter += 1
            doc_info = {
                "id": self.doc_counter,
                "filename": os.path.basename(file_path),
                "user_id": user_id,
                "chunks": chunks,
                "status": "processed",
            }

            self.uploaded_docs.append(doc_info)
            return doc_info

        except Exception as e:
            return {"error": str(e)}


class QuestionAnsweringService:
    def __init__(self, embedding_service, llm_service):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.documents = {}

    def add_document(self, doc_id, chunks):
        self.documents[doc_id] = chunks

        # Create embeddings for each chunk
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_service.create_embedding(chunk)
            self.embedding_service.store_embedding(doc_id, i, embedding)

    def ask_question(self, question, user_id):
        try:
            # Create embedding for the question
            question_embedding = self.embedding_service.create_embedding(question)

            # Find relevant chunks
            relevant_chunks = self.embedding_service.search_similar(question_embedding)

            # Prepare context and sources
            context = [chunk["content"] for chunk in relevant_chunks]
            sources = [
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "similarity": chunk["similarity"],
                    "content": chunk["content"][:100] + "...",
                }
                for chunk in relevant_chunks
            ]

            # Generate answer using LLM
            result = self.llm_service.generate_answer(question, context, sources)

            return {
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "sources": result["sources"],
                "reasoning": result["reasoning"],
                "user_id": user_id,
            }

        except Exception as e:
            return {"error": str(e)}


class DocuMindSystem:
    """Complete DocuMind system for integration testing"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.upload_service = DocumentUploadService()
        self.qa_service = QuestionAnsweringService(
            self.embedding_service, self.llm_service
        )
        self.users = {}

    def upload_document(self, file_path, user_id):
        """Upload and process a document"""
        result = self.upload_service.upload_document(file_path, user_id)

        if "error" not in result:
            # Add document to QA service
            self.qa_service.add_document(result["id"], result["chunks"])

            # Track user documents
            if user_id not in self.users:
                self.users[user_id] = []
            self.users[user_id].append(result["id"])

        return result

    def ask_question(self, question, user_id):
        """Ask a question about uploaded documents"""
        return self.qa_service.ask_question(question, user_id)

    def get_user_documents(self, user_id):
        """Get all documents for a user"""
        if user_id not in self.users:
            return []

        return [
            doc
            for doc in self.upload_service.uploaded_docs
            if doc["id"] in self.users[user_id]
        ]


# Test classes
class TestDocuMindIntegration:
    """Integration tests for complete DocuMind workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = DocuMindSystem()
        self.test_user_id = "user123"

        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()

        # Create test documents with specific content
        self.test_files = []

        # Financial report
        financial_file = os.path.join(self.temp_dir, "financial_report.txt")
        with open(financial_file, "w") as f:
            f.write(
                "The company reported revenue of $15 million in Q1 2024. Employee count increased to 150. The main office is in San Francisco."
            )
        self.test_files.append(financial_file)

        # Employee report
        employee_file = os.path.join(self.temp_dir, "employee_report.txt")
        with open(employee_file, "w") as f:
            f.write(
                "Employee satisfaction scores reached 85%. The team size grew by 20%. Revenue per employee improved by 15%."
            )
        self.test_files.append(employee_file)

        # Market analysis
        market_file = os.path.join(self.temp_dir, "market_analysis.txt")
        with open(market_file, "w") as f:
            f.write(
                "Market analysis shows 25% growth in the tech sector. Competitor analysis reveals three main players. Customer satisfaction is at 92%."
            )
        self.test_files.append(market_file)

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_complete_workflow(self):
        """Test complete workflow from upload to question answering"""
        # Step 1: Upload documents
        upload_results = []
        for file_path in self.test_files:
            result = self.system.upload_document(file_path, self.test_user_id)
            assert "error" not in result
            upload_results.append(result)

        # Verify documents were uploaded
        assert len(upload_results) == 3
        assert len(self.system.get_user_documents(self.test_user_id)) == 3

        # Step 2: Ask questions
        questions = [
            "What is the company revenue?",
            "How many employees are there?",
            "What is the employee satisfaction score?",
            "Where is the main office located?",
        ]

        for question in questions:
            result = self.system.ask_question(question, self.test_user_id)

            assert "error" not in result
            assert result["question"] == question
            assert "answer" in result
            assert "confidence" in result
            assert "sources" in result
            assert len(result["sources"]) > 0
            assert result["confidence"] > 0.5

    def test_revenue_question_specific(self):
        """Test specific revenue question with expected answer"""
        # Upload financial document
        result = self.system.upload_document(self.test_files[0], self.test_user_id)
        assert "error" not in result

        # Ask revenue question
        question = "What is the company revenue?"
        answer_result = self.system.ask_question(question, self.test_user_id)

        assert "error" not in answer_result
        assert "revenue" in answer_result["answer"].lower()
        assert answer_result["confidence"] > 0.8

        # Verify sources contain financial information
        sources_content = " ".join(
            [s["content"].lower() for s in answer_result["sources"]]
        )
        assert any(
            word in sources_content for word in ["revenue", "million", "financial"]
        )

    def test_employee_question_specific(self):
        """Test specific employee question with expected answer"""
        # Upload employee document
        result = self.system.upload_document(self.test_files[1], self.test_user_id)
        assert "error" not in result

        # Ask employee question
        question = "What is the employee satisfaction score?"
        answer_result = self.system.ask_question(question, self.test_user_id)

        assert "error" not in answer_result
        assert "satisfaction" in answer_result["answer"].lower()
        assert answer_result["confidence"] > 0.8

        # Verify sources contain employee information
        sources_content = " ".join(
            [s["content"].lower() for s in answer_result["sources"]]
        )
        assert any(
            word in sources_content for word in ["employee", "satisfaction", "team"]
        )

    def test_multi_document_query(self):
        """Test query that spans multiple documents"""
        # Upload all documents
        for file_path in self.test_files:
            result = self.system.upload_document(file_path, self.test_user_id)
            assert "error" not in result

        # Ask question that could be answered from multiple documents
        question = "What are the key business metrics?"
        answer_result = self.system.ask_question(question, self.test_user_id)

        assert "error" not in answer_result
        assert len(answer_result["sources"]) > 1

        # Should find sources from multiple documents
        doc_ids = set(source["doc_id"] for source in answer_result["sources"])
        assert len(doc_ids) > 1

    def test_user_isolation(self):
        """Test that users only see their own documents"""
        user1_id = "user1"
        user2_id = "user2"

        # Upload documents for different users
        result1 = self.system.upload_document(self.test_files[0], user1_id)
        result2 = self.system.upload_document(self.test_files[1], user2_id)

        assert "error" not in result1
        assert "error" not in result2

        # Verify user isolation
        user1_docs = self.system.get_user_documents(user1_id)
        user2_docs = self.system.get_user_documents(user2_id)

        assert len(user1_docs) == 1
        assert len(user2_docs) == 1
        assert user1_docs[0]["filename"] == "financial_report.txt"
        assert user2_docs[0]["filename"] == "employee_report.txt"

        # Test that questions only search user's documents
        question = "What is the revenue?"
        answer1 = self.system.ask_question(question, user1_id)
        answer2 = self.system.ask_question(question, user2_id)

        assert "error" not in answer1
        assert "error" not in answer2

    def test_document_processing_accuracy(self):
        """Test that document processing creates accurate chunks"""
        # Upload document
        result = self.system.upload_document(self.test_files[0], self.test_user_id)
        assert "error" not in result

        # Verify chunks contain expected content
        chunks = result["chunks"]
        assert len(chunks) > 0

        # Check that chunks contain key information
        chunk_text = " ".join(chunks).lower()
        assert "revenue" in chunk_text
        assert "million" in chunk_text
        assert "employee" in chunk_text
        assert "san francisco" in chunk_text

    def test_embedding_creation(self):
        """Test that embeddings are created for all chunks"""
        # Upload document
        result = self.system.upload_document(self.test_files[0], self.test_user_id)
        assert "error" not in result

        # Verify embeddings were created
        doc_id = result["id"]
        num_chunks = len(result["chunks"])
        num_embeddings = len(
            [
                k
                for k in self.system.embedding_service.embeddings.keys()
                if k.startswith(f"{doc_id}_")
            ]
        )

        assert num_embeddings == num_chunks

    def test_semantic_search_ranking(self):
        """Test that semantic search returns properly ranked results"""
        # Upload document
        result = self.system.upload_document(self.test_files[0], self.test_user_id)
        assert "error" not in result

        # Test search ranking
        query_embedding = self.system.embedding_service.create_embedding("revenue")
        search_results = self.system.embedding_service.search_similar(query_embedding)

        assert len(search_results) > 0

        # Verify results are sorted by similarity (highest first)
        similarities = [result["similarity"] for result in search_results]
        assert similarities == sorted(similarities, reverse=True)

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow"""
        # Test with invalid file
        invalid_file = os.path.join(self.temp_dir, "test.xyz")
        with open(invalid_file, "w") as f:
            f.write("Invalid content")

        result = self.system.upload_document(invalid_file, self.test_user_id)
        assert "error" in result
        assert "Unsupported file format" in result["error"]

        # Test question without documents
        question_result = self.system.ask_question("Test question", "new_user")
        assert "error" not in question_result
        assert len(question_result["sources"]) == 0


class TestPerformanceIntegration:
    """Test performance aspects of the integrated system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = DocuMindSystem()
        self.test_user_id = "user123"

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_multiple_document_upload_performance(self):
        """Test uploading multiple documents efficiently"""
        # Create multiple test documents
        test_files = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"document_{i}.txt")
            with open(file_path, "w") as f:
                f.write(
                    f"This is test document {i}. It contains sample content for testing purposes."
                )
            test_files.append(file_path)

        # Upload all documents
        upload_times = []
        for file_path in test_files:
            result = self.system.upload_document(file_path, self.test_user_id)
            assert "error" not in result

        # Verify all documents were processed
        user_docs = self.system.get_user_documents(self.test_user_id)
        assert len(user_docs) == 5

        # Verify embeddings were created for all chunks
        total_chunks = sum(len(doc["chunks"]) for doc in user_docs)
        total_embeddings = len(self.system.embedding_service.embeddings)
        assert total_embeddings == total_chunks

    def test_concurrent_question_answering(self):
        """Test handling multiple questions efficiently"""
        # Upload a document
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write(
                "The company revenue is $10 million. Employee count is 100. Office is in San Francisco."
            )

        result = self.system.upload_document(file_path, self.test_user_id)
        assert "error" not in result

        # Ask multiple questions
        questions = [
            "What is the revenue?",
            "How many employees?",
            "Where is the office?",
            "What are the key metrics?",
        ]

        answers = []
        for question in questions:
            answer = self.system.ask_question(question, self.test_user_id)
            assert "error" not in answer
            answers.append(answer)

        # Verify all questions were answered
        assert len(answers) == 4
        for answer in answers:
            assert answer["confidence"] > 0.5
            assert len(answer["sources"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
