"""
Question Answering Service for DocuMind AI Assistant with PostgreSQL pgvector
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models import (
    Question,
    DocumentChunk,
    Document,
    SourceCitation,
    QuestionSource,
    SearchResult,
)
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.config import settings
import math


class QuestionAnsweringService:
    """Main question answering service with vector database support"""

    def __init__(self, embedding_service: EmbeddingService, llm_service: LLMService):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.max_concurrent_queries = settings.max_concurrent_queries

    async def ask_question(
        self, question: str, user_id: int, db: Session
    ) -> Dict[str, Any]:
        """Process a question and return answer with citations using vector search"""
        start_time = time.time()

        try:
            # Validate question
            if not question or not question.strip():
                return {
                    "error": "Question cannot be empty",
                    "processing_time": time.time() - start_time,
                }

            if len(question) > 1000:
                return {
                    "error": "Question too long (max 1000 characters)",
                    "processing_time": time.time() - start_time,
                }

            # Create embedding for question
            question_embedding = self.embedding_service.create_embedding(question)
            print(
                f"Created question embedding with {len(question_embedding)} dimensions"
            )

            # Search for relevant chunks using vector similarity
            search_results = self.embedding_service.search_similar(
                db, question_embedding, user_id=user_id, top_k=5
            )
            print(f"Found {len(search_results)} relevant chunks")

            if not search_results:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information in your documents to answer this question. Please try rephrasing your question or upload more documents.",
                    "confidence": 0.0,
                    "sources": [],
                    "reasoning": "No relevant document chunks found using semantic search.",
                    "processing_time": time.time() - start_time,
                }

            # Prepare context and sources for LLM
            chunk_contents = []
            sources = []

            for result in search_results:
                # Verify user has access to this document
                document = (
                    db.query(Document)
                    .filter(
                        Document.id == result["doc_id"],
                        Document.user_id == user_id,
                    )
                    .first()
                )

                if document:
                    chunk_contents.append(result["content"])
                    sources.append(
                        {
                            "doc_id": result["doc_id"],
                            "chunk_id": result["chunk_id"],
                            "similarity": result["similarity"],
                            "content": (
                                result["content"][:200] + "..."
                                if len(result["content"]) > 200
                                else result["content"]
                            ),
                            "filename": result["filename"],
                            "chunk_index": result["chunk_index"],
                        }
                    )

            if not chunk_contents:
                return {
                    "question": question,
                    "answer": "I couldn't find any accessible information in your documents to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "reasoning": "No accessible document chunks found.",
                    "processing_time": time.time() - start_time,
                }

            # Generate answer using LLM with retrieved context
            llm_result = self.llm_service.generate_answer(
                question, chunk_contents, sources
            )

            # If LLM failed and returned a mock answer, do not save, return error
            if (
                llm_result.get("model") == "mock"
                or "fallback" in llm_result.get("reasoning", "").lower()
            ):
                return {
                    "error": "Sorry, I couldn't process your question right now. Please try again later."
                }

            # Validate answer grounding
            grounding_result = self.llm_service.validate_answer_grounding(
                llm_result["answer"], sources
            )

            # Detect hallucination
            hallucination_result = self.llm_service.detect_hallucination(
                llm_result["answer"], sources
            )

            # Adjust confidence based on validation
            adjusted_confidence = self._adjust_confidence(
                llm_result["confidence"], grounding_result, hallucination_result
            )

            # Convert sources to response format
            response_sources = []
            for source in sources:
                response_sources.append(
                    SourceCitation(
                        chunk_id=source["chunk_id"],
                        document_id=source["doc_id"],
                        similarity_score=source["similarity"],
                        content=source["content"],
                        filename=source["filename"],
                    )
                )

            # Save question to database
            db_question = Question(
                user_id=user_id,
                question_text=question,
                answer_text=llm_result["answer"],
                confidence_score=self._sanitize_float(adjusted_confidence),
                processing_time=self._sanitize_float(time.time() - start_time),
            )
            db.add(db_question)
            db.flush()  # Get the ID

            # Save sources to database
            for source in sources:
                db_source = QuestionSource(
                    question_id=db_question.id,
                    chunk_id=source["chunk_id"],
                    similarity_score=self._sanitize_float(source["similarity"]),
                )
                db.add(db_source)

            db.commit()

            response = {
                "question": question,
                "answer": llm_result["answer"],
                "confidence": self._sanitize_float(adjusted_confidence),
                "sources": response_sources,
                "reasoning": llm_result["reasoning"],
                "processing_time": self._sanitize_float(time.time() - start_time),
                "grounding_score": self._sanitize_float(grounding_result["score"]),
                "hallucination_risk": self._sanitize_float(
                    hallucination_result["hallucination_risk"]
                ),
                "model": llm_result["model"],
                "search_results_count": len(search_results),
            }
            return self._sanitize_json(response)

        except Exception as e:
            db.rollback()
            print(f"Error in ask_question: {e}")
            return {
                "error": f"Error processing question: {str(e)}",
                "processing_time": time.time() - start_time,
            }

    def _adjust_confidence(
        self,
        base_confidence: float,
        grounding_result: Dict[str, Any],
        hallucination_result: Dict[str, Any],
    ) -> float:
        """Adjust confidence based on validation results"""
        try:
            adjusted_confidence = base_confidence

            # Adjust based on grounding score
            grounding_score = grounding_result.get("score", 0.5)
            if grounding_score < 0.7:
                adjusted_confidence *= 0.8
            elif grounding_score > 0.9:
                adjusted_confidence *= 1.1

            # Adjust based on hallucination risk
            hallucination_risk = hallucination_result.get("hallucination_risk", "low")
            if hallucination_risk == "high":
                adjusted_confidence *= 0.6
            elif hallucination_risk == "medium":
                adjusted_confidence *= 0.8

            # Ensure confidence stays within bounds
            return max(0.0, min(1.0, adjusted_confidence))

        except Exception as e:
            print(f"Error adjusting confidence: {e}")
            return base_confidence

    async def batch_ask_questions(
        self, questions: List[str], user_id: int, db: Session
    ) -> List[Dict[str, Any]]:
        """Process multiple questions concurrently"""
        try:
            semaphore = asyncio.Semaphore(self.max_concurrent_queries)

            async def process_question(question: str) -> Dict[str, Any]:
                async with semaphore:
                    return await self.ask_question(question, user_id, db)

            # Process questions concurrently
            tasks = [process_question(question) for question in questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "question": questions[i],
                            "error": f"Error processing question: {str(result)}",
                            "confidence": 0.0,
                            "sources": [],
                        }
                    )
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            return [{"error": f"Batch processing failed: {str(e)}"}]

    def _sanitize_float(self, value):
        try:
            f = float(value)
            return f if math.isfinite(f) else 0.0
        except Exception:
            return 0.0

    def _sanitize_json(self, obj):
        # Recursively sanitize all float values in a dict/list
        if isinstance(obj, dict):
            return {k: self._sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_json(v) for v in obj]
        elif isinstance(obj, float):
            return self._sanitize_float(obj)
        else:
            return obj

    def get_user_question_history(
        self,
        user_id: int,
        db: Session,
        limit: int = 50,
        page: int = 1,
        page_size: int = 10,
    ) -> dict:
        """Get user's question history with pagination"""
        try:
            query = db.query(Question).filter(Question.user_id == user_id)
            total = query.count()
            questions = (
                query.order_by(Question.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

            history = []
            for question in questions:
                # Get sources for this question
                sources = (
                    db.query(QuestionSource)
                    .filter(QuestionSource.question_id == question.id)
                    .all()
                )

                source_list = []
                for source in sources:
                    # Get chunk and document info
                    chunk = (
                        db.query(DocumentChunk)
                        .filter(DocumentChunk.id == source.chunk_id)
                        .first()
                    )
                    if chunk:
                        document = (
                            db.query(Document)
                            .filter(Document.id == chunk.document_id)
                            .first()
                        )
                        if document:
                            source_list.append(
                                {
                                    "chunk_id": source.chunk_id,
                                    "document_id": chunk.document_id,
                                    "similarity_score": self._sanitize_float(
                                        source.similarity_score
                                    ),
                                    "content": (
                                        chunk.content[:200] + "..."
                                        if len(chunk.content) > 200
                                        else chunk.content
                                    ),
                                    "filename": document.filename,
                                }
                            )

                history.append(
                    {
                        "id": question.id,
                        "question": question.question_text,
                        "answer": question.answer_text,
                        "confidence": self._sanitize_float(question.confidence_score),
                        "processing_time": self._sanitize_float(
                            question.processing_time
                        ),
                        "created_at": question.created_at.isoformat(),
                        "sources": source_list,
                    }
                )

            return {"history": history, "total": total}

        except Exception as e:
            print(f"Error getting question history: {e}")
            return {"history": [], "total": 0}

    def get_question_analytics(self, user_id: int, db: Session) -> Dict[str, Any]:
        """Get analytics for user's questions"""
        try:
            # Get total questions
            total_questions = (
                db.query(Question).filter(Question.user_id == user_id).count()
            )

            # Get average confidence
            avg_confidence = (
                db.query(func.avg(Question.confidence_score))
                .filter(Question.user_id == user_id)
                .scalar()
            ) or 0.0

            # Get average processing time
            avg_processing_time = (
                db.query(func.avg(Question.processing_time))
                .filter(Question.user_id == user_id)
                .scalar()
            ) or 0.0

            # Get questions by confidence level
            high_confidence = (
                db.query(Question)
                .filter(Question.user_id == user_id, Question.confidence_score >= 0.8)
                .count()
            )

            medium_confidence = (
                db.query(Question)
                .filter(
                    Question.user_id == user_id,
                    Question.confidence_score >= 0.5,
                    Question.confidence_score < 0.8,
                )
                .count()
            )

            low_confidence = (
                db.query(Question)
                .filter(Question.user_id == user_id, Question.confidence_score < 0.5)
                .count()
            )

            return {
                "total_questions": total_questions,
                "average_confidence": float(avg_confidence),
                "average_processing_time": float(avg_processing_time),
                "confidence_distribution": {
                    "high": high_confidence,
                    "medium": medium_confidence,
                    "low": low_confidence,
                },
            }

        except Exception as e:
            print(f"Error getting question analytics: {e}")
            return {}

    def validate_question_quality(self, question: str) -> Dict[str, Any]:
        """Validate question quality and provide recommendations"""
        try:
            issues = []
            recommendations = []

            # Check question length
            if len(question) < 10:
                issues.append("Question is too short")
                recommendations.append("Try to be more specific and detailed")

            if len(question) > 500:
                issues.append("Question is too long")
                recommendations.append("Try to be more concise")

            # Check for common issues
            if question.lower().count("?") == 0:
                issues.append("Question doesn't end with a question mark")
                recommendations.append("End your question with a question mark")

            # Check for vague words
            vague_words = ["thing", "stuff", "something", "anything", "everything"]
            if any(word in question.lower() for word in vague_words):
                issues.append("Question contains vague terms")
                recommendations.append("Be more specific in your question")

            # Check for multiple questions
            if question.count("?") > 1:
                issues.append("Question contains multiple questions")
                recommendations.append("Ask one question at a time")

            quality_score = max(0.0, 1.0 - (len(issues) * 0.2))

            return {
                "quality_score": quality_score,
                "issues": issues,
                "recommendations": recommendations,
                "is_valid": len(issues) == 0,
            }

        except Exception as e:
            print(f"Error validating question quality: {e}")
            return {
                "quality_score": 0.0,
                "issues": ["Error validating question"],
                "recommendations": ["Please try again"],
                "is_valid": False,
            }

    def search_similar_questions(
        self, question: str, user_id: int, db: Session, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar questions using vector similarity (in-memory, numpy only)"""
        try:
            # Create embedding for the question
            question_embedding = self.embedding_service.create_embedding(question)

            # Get all questions for the user
            user_questions = (
                db.query(Question).filter(Question.user_id == user_id).all()
            )

            if not user_questions:
                return []

            # Calculate similarities using numpy
            import numpy as np

            similarities = []
            for q in user_questions:
                q_embedding = self.embedding_service.create_embedding(q.question_text)
                # Calculate cosine similarity using numpy
                a = np.array(question_embedding)
                b = np.array(q_embedding)
                similarity = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                )
                similarities.append(
                    {
                        "question_id": q.id,
                        "question_text": q.question_text,
                        "answer_text": q.answer_text,
                        "similarity": similarity,
                        "created_at": q.created_at.isoformat(),
                    }
                )

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:limit]

        except Exception as e:
            print(f"Error searching similar questions: {e}")
            return []
