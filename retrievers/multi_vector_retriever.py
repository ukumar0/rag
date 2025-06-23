# In real systems, this would be token-level or phrase-level embedding.

from openai import OpenAI
import numpy as np
import random
from typing import List, Dict, Any
from config import OPENAI_API_KEY

class MultiVectorRetriever:
    def __init__(self, model_name="text-embedding-3-small"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.documents = []
        self.document_metadata = []
        self.embeddings = None

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

    def _create_multi_vector_variations(self, texts: List[str]) -> np.ndarray:
        """Create multiple vector representations by varying input"""
        all_embeddings = []
        
        # Original embeddings
        original_embeddings = self._get_embeddings(texts)
        all_embeddings.append(original_embeddings)
        
        # Add variations by modifying text slightly
        variations = []
        for text in texts:
            # Add question context
            variations.append(f"Question context: {text}")
        
        variation_embeddings = self._get_embeddings(variations)
        all_embeddings.append(variation_embeddings)
        
        # Average the embeddings for multi-vector effect
        averaged_embeddings = np.mean(all_embeddings, axis=0)
        
        # Add small random noise to simulate true multi-vector approach
        noise = np.random.normal(0, 0.01, averaged_embeddings.shape)
        return averaged_embeddings + noise

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
        
        # Create multi-vector embeddings
        self.embeddings = self._create_multi_vector_variations(texts)

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
        if self.embeddings is None:
            raise ValueError("No documents have been indexed yet")
            
        if not self.documents:
            raise ValueError("No documents available")
            
        # Encode query with variations and compute similarities
        query_variations = [query, f"Question context: {query}"]
        query_embeddings = self._get_embeddings(query_variations)
        query_vec = np.mean(query_embeddings, axis=0)
        
        scores = [float(np.dot(query_vec, doc_vec)) for doc_vec in self.embeddings]
        
        # Get top k indices
        top_k = min(top_k, len(self.documents))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Format results
        results = []
        for i in top_indices:
            if 0 <= i < len(self.documents):
                result = {
                    'content': self.documents[i],
                    'metadata': self.document_metadata[i],
                    'score': scores[i]
                }
                results.append(result)
            
        return results
