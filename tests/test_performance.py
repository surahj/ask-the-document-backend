import pytest
import time
import tempfile
import os
import threading
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures


# Mock classes for performance testing
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def start_timer(self, operation_name):
        """Start timing an operation"""
        self.metrics[operation_name] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
        }

    def end_timer(self, operation_name):
        """End timing an operation"""
        if operation_name in self.metrics:
            self.metrics[operation_name]["end_time"] = time.time()
            self.metrics[operation_name]["duration"] = (
                self.metrics[operation_name]["end_time"]
                - self.metrics[operation_name]["start_time"]
            )

    def get_duration(self, operation_name):
        """Get duration of an operation"""
        if operation_name in self.metrics:
            return self.metrics[operation_name].get("duration", 0)
        return 0

    def get_all_metrics(self):
        """Get all performance metrics"""
        return self.metrics


class OptimizedDocumentProcessor:
    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt", ".md"]
        self.chunk_size = 1000  # characters per chunk
        self.overlap = 200  # character overlap between chunks

    def validate_document(self, file_path):
        """Validate document format and size"""
        if not any(file_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(
                f"Unsupported file format. Supported: {self.supported_formats}"
            )

        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit for performance testing
            raise ValueError("File too large. Maximum size: 100MB")

        return True

    def process_document_optimized(self, file_path):
        """Process document with optimized chunking"""
        with open(file_path, "r") as f:
            content = f.read()

        # Optimized chunking with overlap
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind(".")
                if (
                    last_period > self.chunk_size * 0.7
                ):  # If period is in last 30% of chunk
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks


class OptimizedEmbeddingService:
    def __init__(self):
        self.embeddings = {}
        self.embedding_cache = {}
        self.batch_size = 10

    def create_embedding(self, text):
        """Create embedding for text with caching"""
        # Simple hash for caching
        text_hash = hash(text) % 1000000

        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Mock embedding creation
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.embedding_cache[text_hash] = embedding
        return embedding

    def create_embeddings_batch(self, texts):
        """Create embeddings for multiple texts in batch"""
        embeddings = []
        for text in texts:
            embedding = self.create_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def store_embedding(self, doc_id, chunk_id, embedding):
        """Store embedding in vector database"""
        key = f"{doc_id}_{chunk_id}"
        self.embeddings[key] = embedding

    def search_similar_optimized(self, query_embedding, top_k=5):
        """Optimized similarity search"""
        # Mock optimized search
        results = []
        for key, embedding in self.embeddings.items():
            doc_id, chunk_id = key.split("_")
            # Mock similarity calculation
            similarity = 0.9 - (0.1 * int(chunk_id))
            results.append(
                {
                    "doc_id": int(doc_id),
                    "chunk_id": int(chunk_id),
                    "similarity": similarity,
                    "content": f"Content from document {doc_id}, chunk {chunk_id}",
                }
            )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class OptimizedLLMService:
    def __init__(self):
        self.model = "gpt-4"
        self.response_cache = {}
        self.max_context_length = 4000

    def generate_answer_optimized(self, question, context_chunks, sources):
        """Generate answer with context optimization"""
        # Create cache key
        cache_key = hash(question + str(context_chunks)) % 1000000

        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # Optimize context length
        optimized_context = self._optimize_context(context_chunks)

        # Mock LLM response
        response = {
            "answer": "This is a comprehensive answer based on the optimized context.",
            "confidence": 0.92,
            "sources": sources,
            "reasoning": f"Generated answer based on {len(optimized_context)} optimized context chunks.",
        }

        # Cache response
        self.response_cache[cache_key] = response
        return response

    def _optimize_context(self, context_chunks):
        """Optimize context to fit within token limits"""
        total_length = sum(len(chunk) for chunk in context_chunks)

        if total_length <= self.max_context_length:
            return context_chunks

        # Truncate context if too long
        optimized = []
        current_length = 0

        for chunk in context_chunks:
            if current_length + len(chunk) <= self.max_context_length:
                optimized.append(chunk)
                current_length += len(chunk)
            else:
                break

        return optimized


class PerformanceOptimizedSystem:
    """Performance-optimized DocuMind system"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.processor = OptimizedDocumentProcessor()
        self.embedding_service = OptimizedEmbeddingService()
        self.llm_service = OptimizedLLMService()
        self.documents = {}
        self.users = {}

    def upload_document_performance(self, file_path, user_id):
        """Upload document with performance monitoring"""
        self.monitor.start_timer("document_upload")

        try:
            # Validate document
            self.monitor.start_timer("document_validation")
            self.processor.validate_document(file_path)
            self.monitor.end_timer("document_validation")

            # Process document
            self.monitor.start_timer("document_processing")
            chunks = self.processor.process_document_optimized(file_path)
            self.monitor.end_timer("document_processing")

            # Create embeddings
            self.monitor.start_timer("embedding_creation")
            embeddings = self.embedding_service.create_embeddings_batch(chunks)
            self.monitor.end_timer("embedding_creation")

            # Store embeddings
            self.monitor.start_timer("embedding_storage")
            for i, embedding in enumerate(embeddings):
                self.embedding_service.store_embedding(
                    len(self.documents) + 1, i, embedding
                )
            self.monitor.end_timer("embedding_storage")

            # Store document info
            doc_info = {
                "id": len(self.documents) + 1,
                "filename": os.path.basename(file_path),
                "user_id": user_id,
                "chunks": chunks,
                "status": "processed",
            }

            self.documents[doc_info["id"]] = doc_info

            if user_id not in self.users:
                self.users[user_id] = []
            self.users[user_id].append(doc_info["id"])

            self.monitor.end_timer("document_upload")
            return doc_info

        except Exception as e:
            self.monitor.end_timer("document_upload")
            return {"error": str(e)}

    def ask_question_performance(self, question, user_id):
        """Ask question with performance monitoring"""
        self.monitor.start_timer("question_answering")

        try:
            # Create question embedding
            self.monitor.start_timer("question_embedding")
            question_embedding = self.embedding_service.create_embedding(question)
            self.monitor.end_timer("question_embedding")

            # Search for relevant chunks
            self.monitor.start_timer("semantic_search")
            relevant_chunks = self.embedding_service.search_similar_optimized(
                question_embedding
            )
            self.monitor.end_timer("semantic_search")

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

            # Generate answer
            self.monitor.start_timer("llm_generation")
            result = self.llm_service.generate_answer_optimized(
                question, context, sources
            )
            self.monitor.end_timer("llm_generation")

            self.monitor.end_timer("question_answering")

            return {
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "sources": result["sources"],
                "reasoning": result["reasoning"],
                "user_id": user_id,
            }

        except Exception as e:
            self.monitor.end_timer("question_answering")
            return {"error": str(e)}

    def get_performance_metrics(self):
        """Get all performance metrics"""
        return self.monitor.get_all_metrics()


# Test classes
class TestDocumentProcessingPerformance:
    """Test document processing performance"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = PerformanceOptimizedSystem()
        self.test_user_id = "user123"

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_large_document(self, size_mb=10):
        """Create a large document for testing"""
        file_path = os.path.join(self.temp_dir, f"large_document_{size_mb}mb.txt")

        # Create content of specified size
        content_size = size_mb * 1024 * 1024  # Convert MB to bytes
        chunk_size = 1000  # 1KB chunks

        with open(file_path, "w") as f:
            written = 0
            chunk_num = 0
            while written < content_size:
                chunk = f"This is chunk {chunk_num} of the large document. It contains sample content for performance testing. "
                chunk += f"The document is {size_mb}MB in size and contains various types of content. "
                chunk += f"This chunk is approximately 1KB in size and contains meaningful text content. "
                chunk += f"Chunk number {chunk_num} is being written to test document processing performance. "
                chunk += f"The content includes various topics and information for comprehensive testing. "

                f.write(chunk)
                written += len(chunk.encode("utf-8"))
                chunk_num += 1

        return file_path

    @pytest.mark.performance
    def test_large_document_processing(self):
        """Test processing of large documents"""
        # Create 5MB document
        large_file = self.create_large_document(5)

        result = self.system.upload_document_performance(large_file, self.test_user_id)

        assert "error" not in result
        assert result["status"] == "processed"
        assert len(result["chunks"]) > 0

        # Check performance metrics
        metrics = self.system.get_performance_metrics()

        # Document upload should complete within reasonable time
        upload_time = metrics.get("document_upload", {}).get("duration", 0)
        assert upload_time < 30  # Should complete within 30 seconds

        # Individual operations should be fast
        validation_time = metrics.get("document_validation", {}).get("duration", 0)
        processing_time = metrics.get("document_processing", {}).get("duration", 0)
        embedding_time = metrics.get("embedding_creation", {}).get("duration", 0)

        assert validation_time < 1  # Validation should be very fast
        assert processing_time < 10  # Processing should be reasonable
        assert embedding_time < 20  # Embedding creation should be reasonable

    @pytest.mark.performance
    def test_multiple_document_upload(self):
        """Test uploading multiple documents efficiently"""
        # Create multiple documents
        documents = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"document_{i}.txt")
            with open(file_path, "w") as f:
                f.write(
                    f"This is document {i}. It contains sample content for batch processing testing. "
                    * 100
                )
            documents.append(file_path)

        # Upload all documents
        start_time = time.time()
        results = []

        for file_path in documents:
            result = self.system.upload_document_performance(
                file_path, self.test_user_id
            )
            assert "error" not in result
            results.append(result)

        total_time = time.time() - start_time

        # Verify all documents were processed
        assert len(results) == 5
        # Check that documents were added to the system
        assert len(self.system.documents) == 5

        # Total time should be reasonable
        assert total_time < 60  # Should complete within 60 seconds

    @pytest.mark.performance
    def test_concurrent_document_upload(self):
        """Test concurrent document upload"""
        # Create multiple documents
        documents = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"document_{i}.txt")
            with open(file_path, "w") as f:
                f.write(
                    f"This is document {i}. It contains sample content for concurrent processing testing. "
                    * 50
                )
            documents.append(file_path)

        # Upload documents concurrently
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    self.system.upload_document_performance,
                    file_path,
                    self.test_user_id,
                )
                for file_path in documents
            ]

            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        total_time = time.time() - start_time

        # Verify all documents were processed
        assert len(results) == 3
        assert all("error" not in result for result in results)

        # Concurrent upload should be faster than sequential
        assert total_time < 30  # Should complete within 30 seconds


class TestQuestionAnsweringPerformance:
    """Test question answering performance"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = PerformanceOptimizedSystem()
        self.test_user_id = "user123"

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Upload test documents
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"document_{i}.txt")
            with open(file_path, "w") as f:
                f.write(
                    f"This is document {i}. It contains sample content for question answering performance testing. "
                    * 20
                )

            result = self.system.upload_document_performance(
                file_path, self.test_user_id
            )
            assert "error" not in result

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    @pytest.mark.performance
    def test_single_question_performance(self):
        """Test performance of single question answering"""
        question = "What is the content of the documents?"

        result = self.system.ask_question_performance(question, self.test_user_id)

        assert "error" not in result
        assert "answer" in result
        assert "confidence" in result

        # Check performance metrics
        metrics = self.system.get_performance_metrics()

        # Question answering should be fast
        qa_time = metrics.get("question_answering", {}).get("duration", 0)
        assert qa_time < 5  # Should complete within 5 seconds

        # Individual operations should be fast
        embedding_time = metrics.get("question_embedding", {}).get("duration", 0)
        search_time = metrics.get("semantic_search", {}).get("duration", 0)
        llm_time = metrics.get("llm_generation", {}).get("duration", 0)

        assert embedding_time < 1  # Embedding should be very fast
        assert search_time < 2  # Search should be fast
        assert llm_time < 3  # LLM generation should be reasonable

    @pytest.mark.performance
    def test_multiple_questions_performance(self):
        """Test performance of multiple questions"""
        questions = [
            "What is document 0 about?",
            "What is document 1 about?",
            "What is document 2 about?",
            "What are the key topics?",
            "What information is available?",
        ]

        start_time = time.time()
        results = []

        for question in questions:
            result = self.system.ask_question_performance(question, self.test_user_id)
            assert "error" not in result
            results.append(result)

        total_time = time.time() - start_time

        # Verify all questions were answered
        assert len(results) == 5
        assert all("answer" in result for result in results)

        # Total time should be reasonable
        assert total_time < 25  # Should complete within 25 seconds

    @pytest.mark.performance
    def test_concurrent_question_answering(self):
        """Test concurrent question answering"""
        questions = [
            "What is document 0 about?",
            "What is document 1 about?",
            "What is document 2 about?",
        ]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    self.system.ask_question_performance, question, self.test_user_id
                )
                for question in questions
            ]

            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        total_time = time.time() - start_time

        # Verify all questions were answered
        assert len(results) == 3
        assert all("error" not in result for result in results)

        # Concurrent processing should be efficient
        assert total_time < 10  # Should complete within 10 seconds


class TestSystemScalability:
    """Test system scalability"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = PerformanceOptimizedSystem()
        self.test_user_id = "user123"
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    @pytest.mark.performance
    def test_memory_usage_with_large_documents(self):
        """Test memory usage with large documents"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Upload multiple large documents
            for i in range(3):
                file_path = os.path.join(self.temp_dir, f"large_doc_{i}.txt")
                with open(file_path, "w") as f:
                    f.write(f"This is large document {i}. " * 10000)  # ~1MB each

                result = self.system.upload_document_performance(
                    file_path, self.test_user_id
                )
                assert "error" not in result

            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable
            assert memory_increase < 500  # Should not increase by more than 500MB
        except ImportError:
            # Skip memory test if psutil is not available
            pytest.skip("psutil not available for memory monitoring")

    @pytest.mark.performance
    def test_response_time_under_load(self):
        """Test response time under load"""
        # Upload documents
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"doc_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"This is document {i}. " * 100)

            result = self.system.upload_document_performance(
                file_path, self.test_user_id
            )
            assert "error" not in result

        # Ask multiple questions under load
        questions = ["What is the content?"] * 10

        response_times = []
        for question in questions:
            start_time = time.time()
            result = self.system.ask_question_performance(question, self.test_user_id)
            end_time = time.time()

            assert "error" not in result
            response_times.append(end_time - start_time)

        # Response times should be consistent
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        assert avg_response_time < 3  # Average should be under 3 seconds
        assert max_response_time < 5  # Max should be under 5 seconds

    @pytest.mark.performance
    def test_concurrent_user_support(self):
        """Test support for concurrent users"""
        user_ids = [f"user_{i}" for i in range(5)]

        # Each user uploads a document
        for user_id in user_ids:
            file_path = os.path.join(self.temp_dir, f"doc_{user_id}.txt")
            with open(file_path, "w") as f:
                f.write(f"This is document for {user_id}. " * 50)

            result = self.system.upload_document_performance(file_path, user_id)
            assert "error" not in result

        # All users ask questions concurrently
        def user_question_workflow(user_id):
            question = f"What is in the document for {user_id}?"
            result = self.system.ask_question_performance(question, user_id)
            return result

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(user_question_workflow, user_id) for user_id in user_ids
            ]

            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        total_time = time.time() - start_time

        # Verify all users got answers
        assert len(results) == 5
        assert all("error" not in result for result in results)

        # Concurrent user support should be efficient
        assert total_time < 15  # Should complete within 15 seconds


class TestPerformanceOptimizations:
    """Test performance optimizations"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = PerformanceOptimizedSystem()
        self.test_user_id = "user123"
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    @pytest.mark.performance
    def test_embedding_caching(self):
        """Test embedding caching performance"""
        # Create document with repeated content
        file_path = os.path.join(self.temp_dir, "repeated_content.txt")
        repeated_text = "This is repeated content. " * 100
        with open(file_path, "w") as f:
            f.write(repeated_text)

        # Upload document
        result = self.system.upload_document_performance(file_path, self.test_user_id)
        assert "error" not in result

        # Check that embeddings were cached
        cache_size = len(self.system.embedding_service.embedding_cache)
        assert cache_size > 0

        # Ask question to trigger embedding creation
        question = "What is the repeated content?"
        qa_result = self.system.ask_question_performance(question, self.test_user_id)
        assert "error" not in qa_result

        # Cache should be utilized
        new_cache_size = len(self.system.embedding_service.embedding_cache)
        assert new_cache_size >= cache_size

    @pytest.mark.performance
    def test_context_optimization(self):
        """Test context optimization for LLM"""
        # Create document with very long content
        file_path = os.path.join(self.temp_dir, "long_content.txt")
        long_content = "This is a very long sentence. " * 1000
        with open(file_path, "w") as f:
            f.write(long_content)

        # Upload document
        result = self.system.upload_document_performance(file_path, self.test_user_id)
        assert "error" not in result

        # Ask question
        question = "What is the content about?"
        qa_result = self.system.ask_question_performance(question, self.test_user_id)
        assert "error" not in qa_result

        # Check that context was optimized
        metrics = self.system.get_performance_metrics()
        llm_time = metrics.get("llm_generation", {}).get("duration", 0)

        # LLM generation should be fast even with long content
        assert llm_time < 3  # Should complete within 3 seconds


if __name__ == "__main__":
    pytest.main([__file__])
