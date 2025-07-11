"""
LLM service for DocuMind AI Assistant
"""

import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings


class LLMService:
    """LLM service for answer generation"""

    def __init__(self):
        self.model = settings.openai_model
        self.client = None
        self.api_key = settings.openai_api_key

        # Only initialize OpenAI client if API key is provided and not a placeholder
        if self.api_key and self.api_key != "":
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.client = None

    def generate_answer(
        self, question: str, context_chunks: List[str], sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using LLM"""
        try:
            if not self.client:
                return self._generate_mock_answer(question, context_chunks, sources)

            # Prepare context
            context_text = self._prepare_context(context_chunks)

            # Create prompt
            prompt = self._create_prompt(question, context_text)

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
                top_p=0.9,
            )

            answer = response.choices[0].message.content.strip()

            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(answer, sources)

            # Generate reasoning
            reasoning = self._generate_reasoning(question, answer, sources)

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "reasoning": reasoning,
                "model": self.model,
                "tokens_used": (
                    response.usage.total_tokens if hasattr(response, "usage") else 0
                ),
            }

        except Exception as e:
            # Fallback to mock answer on error
            return self._generate_mock_answer(
                question, context_chunks, sources, error=str(e)
            )

    def _prepare_context(self, context_chunks: List[str]) -> str:
        """Prepare context from chunks"""
        if not context_chunks:
            return ""

        # Join chunks with separators
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"Context {i}:\n{chunk}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

            Context:
            {context}

            Question: {question}

            Please provide a clear, accurate answer based only on the information provided in the context.
        """

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are DocuMind AI Assistant, an AI-powered document Q&A system. Your role is to:

            1. Answer questions based ONLY on the provided context
            2. Provide accurate, factual responses
            3. Cite specific information from the context when possible
            4. Acknowledge when information is not available in the context
            5. Be concise but comprehensive
            6. Maintain professional and helpful tone

            Important: Only use information from the provided context. Do not use external knowledge.
        """

    def _calculate_confidence(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the answer"""
        try:
            # Base confidence on source quality
            if not sources:
                return 0.1

            # Calculate average similarity of sources
            avg_similarity = sum(
                source.get("similarity", 0) for source in sources
            ) / len(sources)

            # Adjust based on answer length and quality
            answer_length_factor = min(
                len(answer) / 100, 1.0
            )  # Normalize by expected length

            # Combine factors
            confidence = (avg_similarity * 0.7) + (answer_length_factor * 0.3)

            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5  # Default confidence

    def _generate_reasoning(
        self, question: str, answer: str, sources: List[Dict[str, Any]]
    ) -> str:
        """Generate reasoning for the answer"""
        try:
            if not sources:
                return "No relevant sources found in the provided documents."

            source_count = len(sources)
            avg_similarity = (
                sum(source.get("similarity", 0) for source in sources) / source_count
            )

            reasoning = f"Generated answer based on {source_count} relevant document chunks with average similarity of {avg_similarity:.2f}. "

            if avg_similarity > 0.8:
                reasoning += "High confidence due to strong source relevance."
            elif avg_similarity > 0.6:
                reasoning += "Moderate confidence with good source relevance."
            else:
                reasoning += "Lower confidence due to limited source relevance."

            return reasoning

        except Exception:
            return "Generated answer based on available context."

    def _generate_mock_answer(
        self,
        question: str,
        context_chunks: List[str],
        sources: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate mock answer when LLM is not available"""
        try:
            # Simple answer generation based on keywords
            question_lower = question.lower()
            context_text = " ".join(context_chunks).lower()

            if "revenue" in question_lower:
                answer = "Based on the documents, the revenue information is available in the financial data."
            elif "employee" in question_lower and "satisfaction" in question_lower:
                answer = "The employee satisfaction scores are documented in the employee reports."
            elif "satisfaction" in question_lower:
                answer = "Employee satisfaction information is available in the provided documents."
            elif "employee" in question_lower:
                answer = "Employee-related information can be found in the employee documentation."
            elif "office" in question_lower or "location" in question_lower:
                answer = (
                    "Office location information is available in the company documents."
                )
            else:
                answer = "Based on the provided context, I can provide information about the requested topic."

            if error:
                answer += f" (Note: Using fallback response due to error: {error})"

            # Calculate mock confidence
            confidence = 0.7 if sources else 0.3

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "reasoning": "Generated using fallback response mechanism.",
                "model": "mock",
                "tokens_used": 0,
            }

        except Exception:
            return {
                "answer": "I apologize, but I'm unable to generate an answer at this time.",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "Error in answer generation.",
                "model": "mock",
                "tokens_used": 0,
            }

    def validate_answer_grounding(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that answer is grounded in sources"""
        try:
            if not sources:
                return {
                    "grounded": False,
                    "score": 0.0,
                    "reason": "No sources provided",
                }

            # Extract key facts from answer (simplified)
            answer_words = set(answer.lower().split())

            # Check source content for key words
            source_words = set()
            for source in sources:
                content = source.get("content", "").lower()
                source_words.update(content.split())

            # Calculate overlap
            overlap = len(answer_words.intersection(source_words))
            total_answer_words = len(answer_words)

            if total_answer_words == 0:
                grounding_score = 0.0
            else:
                grounding_score = overlap / total_answer_words

            return {
                "grounded": grounding_score > 0.3,
                "score": grounding_score,
                "overlap_count": overlap,
                "total_words": total_answer_words,
            }

        except Exception:
            return {
                "grounded": False,
                "score": 0.0,
                "reason": "Error in grounding validation",
            }

    def detect_hallucination(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect potential hallucination in answer"""
        try:
            grounding_result = self.validate_answer_grounding(answer, sources)

            # Simple hallucination detection based on grounding
            hallucination_risk = 1.0 - grounding_result["score"]

            risk_level = "low"
            if hallucination_risk > 0.7:
                risk_level = "high"
            elif hallucination_risk > 0.4:
                risk_level = "medium"

            return {
                "hallucination_risk": hallucination_risk,
                "risk_level": risk_level,
                "grounding_score": grounding_result["score"],
                "recommendation": self._get_hallucination_recommendation(risk_level),
            }

        except Exception:
            return {
                "hallucination_risk": 1.0,
                "risk_level": "unknown",
                "grounding_score": 0.0,
                "recommendation": "Unable to assess hallucination risk",
            }

    def _get_hallucination_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on hallucination risk level"""
        recommendations = {
            "low": "Answer appears to be well-grounded in sources.",
            "medium": "Answer has moderate grounding. Consider reviewing sources.",
            "high": "Answer may contain hallucinated content. Verify with original sources.",
        }
        return recommendations.get(risk_level, "Unable to assess answer quality.")
