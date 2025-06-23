from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from typing import List, Dict, Any, Optional
import numpy as np
from .models import Base, Document, Chunk
from config import DATABASE_URL
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(self.engine)
        
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
        
    def document_exists(self, doc_hash: str) -> bool:
        """Check if a document with given hash exists"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.hash == doc_hash).first() is not None
            
    def store_document(self, content: str, metadata: Dict, doc_hash: str) -> Document:
        """Store a new document"""
        with self.get_session() as session:
            document = Document(
                content=content,
                doc_metadata=metadata,
                hash=doc_hash,
                filename=metadata.get('filename')
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
            
    def store_chunks(self, doc_id: int, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """Store document chunks with their embeddings"""
        with self.get_session() as session:
            stored_chunks = []
            for idx, chunk_data in enumerate(chunks):
                chunk = Chunk(
                    document_id=doc_id,
                    content=chunk_data['content'],
                    chunk_metadata=chunk_data.get('metadata', {}),
                    chunk_index=idx
                )
                if 'embedding' in chunk_data:
                    chunk.set_embedding(chunk_data['embedding'])
                session.add(chunk)
                stored_chunks.append(chunk)
            session.commit()
            return stored_chunks
            
    def get_document_by_hash(self, doc_hash: str) -> Optional[Document]:
        """Retrieve a document by its hash"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.hash == doc_hash).first()
            
    def get_chunks_by_doc_id(self, doc_id: int) -> List[Chunk]:
        """Retrieve all chunks for a document"""
        with self.get_session() as session:
            return session.query(Chunk).filter(Chunk.document_id == doc_id).all()
            
    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        Note: This is a simple implementation. For production, consider using pgvector or a dedicated vector DB
        """
        query_embedding_json = query_embedding.tolist()
        
        with self.get_session() as session:
            # Convert embeddings back to arrays and compute cosine similarity
            chunks = session.query(Chunk).filter(Chunk.embedding.isnot(None)).all()
            
            similarities = []
            for chunk in chunks:
                chunk_embedding = np.array(eval(chunk.embedding))
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append((chunk, similarity))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = []
            
            for chunk, similarity in similarities[:top_k]:
                result = chunk.to_dict()
                result['similarity'] = float(similarity)
                results.append(result)
                
            return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self.get_session() as session:
            doc_count = session.query(Document).count()
            chunk_count = session.query(Chunk).count()
            return {
                'documents': doc_count,
                'chunks': chunk_count
            }
            
    def clear_all_data(self):
        """Clear all data from the database (use with caution!)"""
        with self.get_session() as session:
            session.query(Chunk).delete()
            session.query(Document).delete()
            session.commit() 