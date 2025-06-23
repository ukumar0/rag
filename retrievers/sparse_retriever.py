from rank_bm25 import BM25Okapi
import nltk
from typing import List, Dict, Any
nltk.download("punkt", quiet=True)

class SparseRetriever:
    def __init__(self):
        self.tokenized_docs = []
        self.bm25 = None
        self.documents = []
        self.document_metadata = []

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
        
        # Create tokenized documents and BM25 index
        self.tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

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
        if not self.bm25:
            raise ValueError("No documents have been indexed yet")
            
        if not self.documents:
            raise ValueError("No documents available")
            
        # Tokenize query and get scores
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
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
                    'score': float(scores[i])
                }
                results.append(result)
            
        return results
