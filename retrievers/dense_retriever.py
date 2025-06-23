from openai import OpenAI
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from config import OPENAI_API_KEY

class DenseRetriever:
    def __init__(self, model_name="text-embedding-3-small"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.document_metadata = []

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        # Clean and validate input texts
        clean_texts = []
        for text in texts:
            if not text or not isinstance(text, str):
                # Skip empty or non-string inputs
                continue
            
            # Clean the text: remove null bytes but be less aggressive
            cleaned = str(text).replace('\x00', '').strip()
            
            # Skip only if completely empty after basic cleaning
            if not cleaned:
                continue
                
            # Truncate if too long (OpenAI has limits) but use higher limit
            if len(cleaned) > 10000:  # More generous limit
                cleaned = cleaned[:10000] + "..."
            
            clean_texts.append(cleaned)
        
        if not clean_texts:
            # Log what we received to debug
            print(f"DEBUG: Original texts count: {len(texts)}")
            for i, text in enumerate(texts[:3]):  # Show first 3
                print(f"DEBUG: Text {i}: type={type(text)}, len={len(str(text)) if text else 0}, preview='{str(text)[:100]}'")
            raise ValueError("No valid texts to embed after cleaning")
        
        try:
            response = self.client.embeddings.create(
                input=clean_texts,
                model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            # Log the error with more context
            print(f"Error creating embeddings: {e}")
            print(f"Number of texts: {len(clean_texts)}")
            print(f"Sample text (first 200 chars): {clean_texts[0][:200] if clean_texts else 'None'}")
            raise

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents with their metadata.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
        """
        if not documents:
            return
            
        # Extract texts and metadata
        texts = []
        metadata_list = []
        
        for doc in documents:
            if isinstance(doc, dict) and 'content' in doc:
                texts.append(doc['content'])
                metadata_list.append(doc.get('metadata', {}))
            else:
                # Handle legacy format where doc is just text
                texts.append(str(doc))
                metadata_list.append({})
        
        # Store documents and metadata
        self.documents = texts
        self.document_metadata = metadata_list
        
        # Create embeddings and index
        embeddings = self._get_embeddings(texts)
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def build_index(self, texts: List[str]) -> None:
        """Legacy method for backward compatibility"""
        documents = [{'content': text, 'metadata': {}} for text in texts]
        self.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document content, metadata, and relevance score
        """
        if not self.index:
            raise ValueError("No documents have been indexed yet")
            
        if not self.documents:
            raise ValueError("No documents available")
            
        # Encode query and search
        query_embedding = self._get_embeddings([query])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Format results
        results = []
        for i, score in zip(indices[0], scores[0]):
            if 0 <= i < len(self.documents):
                result = {
                    'content': self.documents[i],
                    'metadata': self.document_metadata[i],
                    'score': float(score)
                }
                results.append(result)
            
        return results
