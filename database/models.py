from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List, Dict, Any
import json

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSON, nullable=True)
    filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    hash = Column(String(64), unique=True, nullable=False)
    
    # Relationship with chunks
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.doc_metadata,
            'filename': self.filename,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'hash': self.hash
        }

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)  # Store embeddings as JSON array
    chunk_index = Column(Integer, nullable=False)  # Position in document
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with document
    document = relationship("Document", back_populates="chunks")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'metadata': self.chunk_metadata,
            'embedding': self.embedding,
            'chunk_index': self.chunk_index,
            'created_at': self.created_at.isoformat()
        }
    
    def set_embedding(self, embedding: List[float]) -> None:
        """Store embedding as JSON"""
        self.embedding = json.dumps(embedding.tolist())