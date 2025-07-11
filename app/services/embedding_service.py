"""
Embedding service for DocuMind AI Assistant with SQLite compatibility
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.config import settings
from app.models import DocumentChunk, Document


class EmbeddingService:
    """Embedding service for text vectorization and semantic search using SQLite"""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.similarity_threshold = 0.1  # Force low threshold for testing
        print(
            f"[DEBUG] EmbeddingService initialized with similarity_threshold={self.similarity_threshold}"
        )
        self.top_k = settings.top_k_results
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for given text"""
        try:
            if not text.strip():
                return [0.0] * self.embedding_dimension

            # Create embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            # Ensure we return a plain Python list, not numpy array
            if hasattr(embedding, "tolist"):
                return embedding.tolist()
            elif isinstance(embedding, (list, tuple)):
                return list(embedding)
            else:
                return [float(x) for x in embedding]

        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return zero embedding on error
            return [0.0] * self.embedding_dimension

    def search_similar(
        self,
        db: Session,
        query_embedding: List[float],
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using pgvector"""
        try:
            if not query_embedding:
                return []

            if top_k is None:
                top_k = self.top_k

            # Ensure query_embedding is a plain Python list
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()
            elif not isinstance(query_embedding, list):
                query_embedding = [float(x) for x in query_embedding]

            # PostgreSQL with pgvector only
            from sqlalchemy import text

            sql_query = text(
                """
                SELECT dc.id, dc.content, dc.chunk_index, dc.document_id,
                       d.filename, 1 - (dc.embedding <=> CAST(:query_embedding AS vector)) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE (1 - (dc.embedding <=> CAST(:query_embedding AS vector))) >= :threshold
                AND (:user_id IS NULL OR d.user_id = :user_id)
                ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """
            )

            result = db.execute(
                sql_query,
                {
                    "query_embedding": query_embedding,
                    "threshold": self.similarity_threshold,
                    "user_id": user_id,
                    "top_k": top_k,
                },
            )

            results = []
            for row in result:
                results.append(
                    (
                        DocumentChunk(
                            id=row.id,
                            content=row.content,
                            chunk_index=row.chunk_index,
                            document_id=row.document_id,
                        ),
                        row.filename,
                        row.similarity,
                    )
                )

            # Format results
            formatted_results = []
            for chunk, filename, similarity in results:
                formatted_results.append(
                    {
                        "doc_id": chunk.document_id,
                        "chunk_id": chunk.id,
                        "similarity": float(similarity),
                        "content": chunk.content,
                        "filename": filename,
                        "chunk_index": chunk.chunk_index,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently"""
        try:
            if not texts:
                return []

            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                return []

            # Create embeddings in batch
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False)
            # Ensure we return plain Python lists, not numpy arrays
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            elif isinstance(embeddings, (list, tuple)):
                return [
                    list(emb) if hasattr(emb, "tolist") else emb for emb in embeddings
                ]
            else:
                return [[float(x) for x in emb] for emb in embeddings]

        except Exception as e:
            print(f"Error in batch embedding creation: {e}")
            # Return zero embeddings for all texts on error
            return [[0.0] * self.embedding_dimension] * len(texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension

    def update_similarity_threshold(self, threshold: float) -> bool:
        """Update similarity threshold"""
        try:
            self.similarity_threshold = threshold
            return True
        except Exception:
            return False

    def get_search_stats(
        self, db: Session, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            query = db.query(DocumentChunk)

            if user_id is not None:
                query = query.join(Document).filter(Document.user_id == user_id)

            total_chunks = query.count()
            chunks_with_embeddings = query.filter(
                DocumentChunk.embedding.isnot(None)
            ).count()

            return {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "embedding_coverage": (
                    chunks_with_embeddings / total_chunks if total_chunks > 0 else 0
                ),
                "similarity_threshold": self.similarity_threshold,
                "top_k_results": self.top_k,
            }
        except Exception as e:
            print(f"Error getting search stats: {e}")
            return {}

    def create_embeddings_for_document(
        self, db: Session, document_id: int, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Create and store embeddings for all chunks of a document"""
        try:
            for chunk_data in chunks:
                # Create embedding for chunk content
                embedding = self.create_embedding(chunk_data["content"])

                # Store embedding in database
                chunk = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == document_id,
                        DocumentChunk.chunk_index == chunk_data["chunk_index"],
                    )
                    .first()
                )

                if chunk:
                    embedding_json = json.dumps(embedding)
                    chunk.embedding_json = embedding_json

            db.commit()
            return True

        except Exception as e:
            print(f"Error creating embeddings for document: {e}")
            db.rollback()
            return False

    def delete_embeddings_for_document(self, db: Session, document_id: int) -> bool:
        """Delete all embeddings for a document"""
        try:
            chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .all()
            )

            for chunk in chunks:
                chunk.embedding_json = None

            db.commit()
            return True

        except Exception as e:
            print(f"Error deleting embeddings for document: {e}")
            db.rollback()
            return False

    def _convert_to_numpy_array(self, embedding: List[float]) -> np.ndarray:
        """Convert embedding list to numpy array"""
        try:
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error converting embedding to numpy array: {e}")
            return np.array([0.0] * self.embedding_dimension, dtype=np.float32)
