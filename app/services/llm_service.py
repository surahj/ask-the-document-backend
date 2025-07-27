"""
LLM service for DocuMind AI Assistant
"""

import json
import time
from typing import List, Dict, Any, Optional
from app.config import settings
import requests
import math
import os
from huggingface_hub import InferenceClient


class LLMService:
    """LLM service for answer generation using DeepSeek-V3 via Hugging Face InferenceClient (fireworks-ai)"""

    def __init__(self):
        self.model = settings.llm_model
        self.api_key = settings.huggingface_api_key

        if not self.api_key:
            print("[LLMService] Warning: No Hugging Face API key configured")
            self.client = None
        else:
            try:
                self.client = InferenceClient(
                    provider="fireworks-ai",
                    api_key=self.api_key,
                )
                print(f"[LLMService] Initialized with model: {self.model}")
            except Exception as e:
                print(f"[LLMService] Error initializing client: {e}")
                self.client = None

    def generate_answer(
        self, question: str, context_chunks: List[str], sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.client:
            print("[LLMService] No client available, using fallback response")
            return self._generate_mock_answer(
                question, context_chunks, sources, error="No LLM client available"
            )

        prompt = self._create_prompt(question, self._prepare_context(context_chunks))
        system_prompt = self._get_system_prompt()
        try:
            print(f"[LLMService] Generating answer for question: {question[:50]}...")
            # Use the InferenceClient for text generation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            print(
                f"[LLMService] Successfully generated answer ({len(answer)} characters)"
            )

            confidence = self._sanitize_float(
                self._calculate_confidence(answer, sources)
            )
            reasoning = self._generate_reasoning(question, answer, sources)
            sanitized_sources = self._sanitize_sources(sources)
            response_dict = {
                "answer": answer,
                "confidence": confidence,
                "sources": sanitized_sources,
                "reasoning": reasoning,
                "model": self.model,
                "tokens_used": 0,
            }
            return self._sanitize_json(response_dict)
        except Exception as e:
            print(f"[LLMService] DeepSeek-V3 failed: {e}")
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
        return f"""Based on the following context, please answer the question. 

Context:
{context}

Question: {question}

Instructions:
- If the context contains relevant information, provide an answer based on what you can find.
- If the context is somewhat related but doesn't fully answer the question, provide what you can and note any limitations.
- Only say "Sorry, I could not find the answer in the provided documents" if the context is completely irrelevant to the question.
- Be helpful and informative while being honest about what you can and cannot determine from the context.
- If you find partial information, share it and explain what aspects of the question remain unanswered.

Please provide a clear, accurate answer based on the information provided in the context."""

    def _get_system_prompt(self) -> str:
        """Get improved system prompt for DeepSeek-V3"""
        return (
            "You are DocuMind AI Assistant, an expert document Q&A system. "
            "Your job is to answer user questions using the provided context. "
            "Be helpful and informative while being honest about limitations.\n\n"
            "Instructions:\n"
            "- Use the information from the provided context to answer questions.\n"
            "- Be concise, accurate, and professional.\n"
            "- If you cite information, mention the context number or snippet.\n"
            "- If the context contains relevant information (even if not a perfect match), provide what you can find.\n"
            "- Only say 'Sorry, I could not find the answer in the provided documents' if the context is completely irrelevant to the question.\n"
            "- If the context is somewhat related but doesn't fully answer the question, provide what you can and note any limitations.\n"
            "- Format your answer as HTML for best readability.\n"
            "- Use <h3> for headings, <p> for paragraphs, <ul>/<ol> for lists, and <br> for line breaks.\n"
            "- Each paragraph should be separated by two new lines.\n"
            "Context will be provided below.\n"
        )

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
            sanitized_sources = self._sanitize_sources(sources)
            response_dict = {
                "answer": answer,
                "confidence": self._sanitize_float(confidence),
                "sources": sanitized_sources,
                "reasoning": (
                    "Generated using fallback response mechanism."
                    if not error
                    else f"Generated using fallback response due to error: {error}"
                ),
                "model": "mock",
                "tokens_used": 0,
            }
            return self._sanitize_json(response_dict)

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

    def _sanitize_float(self, value):
        try:
            f = float(value)
            return f if math.isfinite(f) else 0.0
        except Exception:
            return 0.0

    def _sanitize_sources(self, sources):
        # Recursively sanitize all float values in sources
        sanitized = []
        for s in sources:
            s_copy = dict(s)
            for k, v in s_copy.items():
                if isinstance(v, float):
                    s_copy[k] = self._sanitize_float(v)
            sanitized.append(s_copy)
        return sanitized

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
