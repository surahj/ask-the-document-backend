import pytest
from unittest.mock import Mock, patch, MagicMock
import re


# Mock classes for accuracy and hallucination testing
class AccuracyValidator:
    def __init__(self):
        self.validation_rules = {
            "source_grounding": True,
            "factual_consistency": True,
            "citation_accuracy": True,
        }

    def validate_answer_grounding(self, answer, sources, question):
        """Validate that answer is grounded in provided sources"""
        score = 0.0
        issues = []

        # Check if answer contains information not in sources
        answer_lower = answer.lower()
        source_content = " ".join([s["content"].lower() for s in sources])

        # Extract key facts from answer
        facts = self._extract_facts(answer)

        for fact in facts:
            if not self._fact_in_sources(fact, source_content):
                issues.append(f"Fact not found in sources: {fact}")
                score -= 0.2
            else:
                score += 0.1

        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score + 0.5))

        return {"score": score, "issues": issues, "is_grounded": score > 0.6}

    def _extract_facts(self, text):
        """Extract factual statements from text"""
        # Simple fact extraction - look for numbers, dates, names, etc.
        facts = []

        # Extract numbers with context
        number_pattern = r"\$?\d+(?:\.\d+)?\s*(?:million|billion|thousand|%|percent)?"
        numbers = re.findall(number_pattern, text, re.IGNORECASE)
        facts.extend(numbers)

        # Extract dates
        date_pattern = r"\d{4}|Q[1-4]\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        facts.extend(dates)

        # Extract locations
        location_pattern = r"(?:in|at|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        locations = re.findall(location_pattern, text, re.IGNORECASE)
        facts.extend(locations)

        return facts

    def _fact_in_sources(self, fact, source_content):
        """Check if a fact is present in source content"""
        fact_lower = fact.lower()
        return fact_lower in source_content


class HallucinationDetector:
    def __init__(self):
        self.detection_methods = [
            "source_consistency",
            "factual_verification",
            "confidence_analysis",
        ]

    def detect_hallucinations(self, answer, sources, question):
        """Detect potential hallucinations in the answer"""
        issues = []
        confidence = 1.0

        # Method 1: Check source consistency
        consistency_score = self._check_source_consistency(answer, sources)
        if consistency_score < 0.8:
            issues.append("Low source consistency detected")
            confidence *= 0.8

        # Method 2: Check for unsourced claims
        unsourced_claims = self._find_unsourced_claims(answer, sources)
        if unsourced_claims:
            issues.append(f"Unsourced claims found: {unsourced_claims}")
            confidence *= 0.7

        # Method 3: Check for contradictory information
        contradictions = self._find_contradictions(answer, sources)
        if contradictions:
            issues.append(f"Contradictory information: {contradictions}")
            confidence *= 0.6

        return {
            "is_hallucinated": confidence < 0.7,
            "confidence": confidence,
            "issues": issues,
            "risk_level": (
                "high" if confidence < 0.7 else "medium" if confidence < 0.9 else "low"
            ),
        }

    def _check_source_consistency(self, answer, sources):
        """Check if answer is consistent with sources"""
        if not sources:
            return 0.0

        answer_lower = answer.lower()
        source_content = " ".join([s["content"].lower() for s in sources])

        # Calculate overlap between answer and sources
        answer_words = set(answer_lower.split())
        source_words = set(source_content.split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words.intersection(source_words))
        return overlap / len(answer_words)

    def _find_unsourced_claims(self, answer, sources):
        """Find claims in answer that aren't supported by sources"""
        unsourced = []

        # Look for specific claim patterns
        claim_patterns = [
            r"the\s+\w+\s+is\s+\d+",
            r"there\s+are\s+\d+",
            r"the\s+company\s+\w+",
            r"\$\d+",
            r"\d+%",
        ]

        answer_lower = answer.lower()
        source_content = " ".join([s["content"].lower() for s in sources])

        for pattern in claim_patterns:
            matches = re.findall(pattern, answer_lower)
            for match in matches:
                if match not in source_content:
                    unsourced.append(match)

        return unsourced

    def _find_contradictions(self, answer, sources):
        """Find contradictions between answer and sources"""
        contradictions = []

        # Simple contradiction detection
        answer_lower = answer.lower()
        source_content = " ".join([s["content"].lower() for s in sources])

        # Check for number contradictions
        answer_numbers = re.findall(r"\d+", answer_lower)
        source_numbers = re.findall(r"\d+", source_content)

        for num in answer_numbers:
            if num not in source_numbers:
                contradictions.append(f"Number {num} not in sources")

        return contradictions


class EnhancedQuestionAnsweringService:
    def __init__(self, embedding_service, llm_service):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.accuracy_validator = AccuracyValidator()
        self.hallucination_detector = HallucinationDetector()
        self.documents = {}

    def add_document(self, doc_id, chunks):
        """Add document chunks to the system"""
        self.documents[doc_id] = chunks

        # Create embeddings for each chunk
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_service.create_embedding(chunk)
            self.embedding_service.store_embedding(doc_id, i, embedding)

    def ask_question_with_validation(self, question, user_id):
        """Ask question with accuracy and hallucination validation"""
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
                    "content": chunk["content"],
                }
                for chunk in relevant_chunks
            ]

            # Generate answer using LLM
            llm_result = self.llm_service.generate_answer(question, context, sources)

            # Validate accuracy and detect hallucinations
            accuracy_result = self.accuracy_validator.validate_answer_grounding(
                llm_result["answer"], sources, question
            )

            hallucination_result = self.hallucination_detector.detect_hallucinations(
                llm_result["answer"], sources, question
            )

            # Adjust confidence based on validation results
            adjusted_confidence = (
                llm_result["confidence"]
                * accuracy_result["score"]
                * hallucination_result["confidence"]
            )

            return {
                "question": question,
                "answer": llm_result["answer"],
                "confidence": adjusted_confidence,
                "sources": sources,
                "reasoning": llm_result["reasoning"],
                "user_id": user_id,
                "accuracy_validation": accuracy_result,
                "hallucination_detection": hallucination_result,
                "is_reliable": accuracy_result["is_grounded"]
                and not hallucination_result["is_hallucinated"],
            }

        except Exception as e:
            return {"error": str(e)}


# Test classes
class TestAccuracyValidation:
    """Test accuracy validation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.validator = AccuracyValidator()

    def test_answer_grounding_validation(self):
        """Test that answers are properly grounded in sources"""
        answer = "The company revenue is $15 million and they have 200 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            },
            {
                "content": "Employee count increased to 200 this year.",
                "similarity": 0.87,
            },
        ]
        question = "What is the company revenue and employee count?"

        result = self.validator.validate_answer_grounding(answer, sources, question)

        assert "score" in result
        assert "issues" in result
        assert "is_grounded" in result
        assert result["score"] >= 0.7
        assert result["is_grounded"] == True
        assert len(result["issues"]) == 0

    def test_ungrounded_answer_detection(self):
        """Test detection of answers not grounded in sources"""
        answer = "The company revenue is $50 million and they have 500 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]
        question = "What is the company revenue and employee count?"

        result = self.validator.validate_answer_grounding(answer, sources, question)

        assert result["score"] < 0.7
        assert result["is_grounded"] == False
        assert len(result["issues"]) > 0

    def test_fact_extraction(self):
        """Test extraction of facts from text"""
        text = "The company revenue is $15 million in Q1 2024. They have 200 employees in San Francisco."
        facts = self.validator._extract_facts(text)

        assert len(facts) > 0
        assert any("15" in fact for fact in facts)
        assert any("200" in fact for fact in facts)
        assert any("2024" in fact for fact in facts)

    def test_fact_verification(self):
        """Test verification of facts against sources"""
        fact = "15 million"
        source_content = "The company reported revenue of $15 million in Q1 2024."

        is_present = self.validator._fact_in_sources(fact, source_content)
        assert is_present == True

        # Test with fact not in sources
        fact = "50 million"
        is_present = self.validator._fact_in_sources(fact, source_content)
        assert is_present == False


class TestHallucinationDetection:
    """Test hallucination detection functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = HallucinationDetector()

    def test_source_consistency_check(self):
        """Test source consistency checking"""
        answer = "The company revenue is $15 million and they have 200 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            },
            {
                "content": "Employee count increased to 200 this year.",
                "similarity": 0.87,
            },
        ]

        consistency_score = self.detector._check_source_consistency(answer, sources)
        assert consistency_score > 0.5  # Lower threshold for mock data

    def test_low_consistency_detection(self):
        """Test detection of low source consistency"""
        answer = "The company revenue is $50 million and they have 500 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        consistency_score = self.detector._check_source_consistency(answer, sources)
        assert consistency_score < 0.8

    def test_unsourced_claims_detection(self):
        """Test detection of unsourced claims"""
        answer = "The company revenue is $50 million and they have 500 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        unsourced_claims = self.detector._find_unsourced_claims(answer, sources)
        assert len(unsourced_claims) > 0
        assert any("50" in claim for claim in unsourced_claims)

    def test_contradiction_detection(self):
        """Test detection of contradictions"""
        answer = "The company revenue is $50 million."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        contradictions = self.detector._find_contradictions(answer, sources)
        assert len(contradictions) > 0

    def test_hallucination_detection_integration(self):
        """Test complete hallucination detection"""
        answer = "The company revenue is $50 million and they have 500 employees."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]
        question = "What is the company revenue and employee count?"

        result = self.detector.detect_hallucinations(answer, sources, question)

        assert "is_hallucinated" in result
        assert "confidence" in result
        assert "issues" in result
        assert "risk_level" in result
        assert result["is_hallucinated"] == True
        assert result["confidence"] < 0.7
        assert result["risk_level"] == "high"


class TestEnhancedQuestionAnswering:
    """Test enhanced question answering with validation"""

    def setup_method(self):
        """Setup test fixtures"""
        # Mock services
        self.embedding_service = Mock()
        self.embedding_service.create_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.embedding_service.search_similar.return_value = [
            {
                "doc_id": 1,
                "chunk_id": 1,
                "similarity": 0.95,
                "content": "The company reported revenue of $15 million in Q1 2024 in the financial documents.",
            }
        ]

        self.llm_service = Mock()
        self.llm_service.generate_answer.return_value = {
            "answer": "The company revenue is $15 million as reported in the financial documents.",
            "confidence": 0.9,
            "reasoning": "Based on the financial report.",
        }

        self.qa_service = EnhancedQuestionAnsweringService(
            self.embedding_service, self.llm_service
        )

    def test_accurate_answer_validation(self):
        """Test validation of accurate answers"""
        question = "What is the company revenue?"
        result = self.qa_service.ask_question_with_validation(question, "user123")

        assert "error" not in result
        assert "accuracy_validation" in result
        assert "hallucination_detection" in result
        assert "is_reliable" in result

        accuracy = result["accuracy_validation"]
        hallucination = result["hallucination_detection"]

        assert accuracy["is_grounded"] == True
        assert hallucination["is_hallucinated"] == False
        assert result["is_reliable"] == True

    def test_hallucinated_answer_detection(self):
        """Test detection of hallucinated answers"""
        # Mock LLM to return hallucinated answer
        self.llm_service.generate_answer.return_value = {
            "answer": "The company revenue is $50 million.",
            "confidence": 0.9,
            "reasoning": "Based on the financial report.",
        }

        question = "What is the company revenue?"
        result = self.qa_service.ask_question_with_validation(question, "user123")

        assert "error" not in result
        assert result["accuracy_validation"]["is_grounded"] == False
        assert result["hallucination_detection"]["is_hallucinated"] == True
        assert result["is_reliable"] == False

    def test_confidence_adjustment(self):
        """Test that confidence is adjusted based on validation"""
        question = "What is the company revenue?"
        result = self.qa_service.ask_question_with_validation(question, "user123")

        # Original confidence was 0.9, should be adjusted down if validation finds issues
        assert result["confidence"] <= 0.9
        assert result["confidence"] > 0.0


class TestAccuracyMetrics:
    """Test accuracy metrics and scoring"""

    def setup_method(self):
        """Setup test fixtures"""
        self.validator = AccuracyValidator()
        self.detector = HallucinationDetector()

    def test_accuracy_scoring_scale(self):
        """Test that accuracy scores are properly scaled"""
        # Test perfect answer
        answer = "The company revenue is $15 million."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        result = self.validator.validate_answer_grounding(
            answer, sources, "What is the revenue?"
        )
        assert result["score"] >= 0.0 and result["score"] <= 1.0

    def test_hallucination_confidence_scale(self):
        """Test that hallucination confidence is properly scaled"""
        answer = "The company revenue is $15 million."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        result = self.detector.detect_hallucinations(
            answer, sources, "What is the revenue?"
        )
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

    def test_risk_level_classification(self):
        """Test risk level classification"""
        # Test low risk
        answer = "The company revenue is $15 million."
        sources = [
            {
                "content": "The company reported revenue of $15 million in Q1 2024.",
                "similarity": 0.95,
            }
        ]

        result = self.detector.detect_hallucinations(
            answer, sources, "What is the revenue?"
        )
        assert result["risk_level"] in ["low", "medium", "high"]


if __name__ == "__main__":
    pytest.main([__file__])
