from collections import defaultdict
from functools import lru_cache
import hashlib
from typing import List, Dict

class HybridRetriever:
    def __init__(self, dense, sparse, multi, cache_size=1000):
        self.dense = dense
        self.sparse = sparse
        self.multi = multi
        self.weights = {'dense': 0.6, 'sparse': 0.3, 'multi': 0.1}
        self.conversation_history = []
        self._cached_retrieve = lru_cache(maxsize=cache_size)(self._retrieve_impl)
        self.documents = []
        self.document_metadata = []

    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Index documents in all retrievers.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
        """
        self.documents = documents
        self.dense.index_documents(documents)
        self.sparse.index_documents(documents)
        self.multi.index_documents(documents)

    def retrieve(self, query, top_k=5, conversation_context=None):
        # Update conversation history
        if conversation_context:
            self.conversation_history.append(conversation_context)
            self.conversation_history = self.conversation_history[-5:]  # Keep last 5 turns
        
        # Create cache key from query and conversation history
        cache_key = self._create_cache_key(query, conversation_context)
        
        # Get results from cache or compute
        return self._cached_retrieve(cache_key, query, top_k)

    def _create_cache_key(self, query, context=None):
        key_content = query
        if context:
            key_content += "||" + str(context)
        if self.conversation_history:
            key_content += "||" + "||".join(self.conversation_history[-3:])
        return hashlib.md5(key_content.encode()).hexdigest()

    def _retrieve_impl(self, cache_key, query, top_k):
        # Enhance query with conversation history if available
        enhanced_query = self._enhance_query_with_history(query)
        
        # Get results from each retriever
        results = {
            'dense': self.dense.retrieve(enhanced_query, top_k),
            'sparse': self.sparse.retrieve(enhanced_query, top_k),
            'multi': self.multi.retrieve(enhanced_query, top_k)
        }

        # Combine scores using reciprocal rank fusion
        fusion_scores = defaultdict(float)
        doc_metadata = {}
        
        for method, docs in results.items():
            for rank, doc in enumerate(docs):
                content = doc['content']
                metadata = doc['metadata']
                score = doc['score']
                
                # Store metadata for this document
                doc_metadata[content] = metadata
                
                # Calculate fusion score
                fusion_scores[content] += self.weights[method] * score / (rank + 1)

        # Sort documents by fusion score
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k documents with their metadata and scores
        return [
            {
                'content': content,
                'metadata': doc_metadata[content],
                'score': score
            }
            for content, score in sorted_docs[:top_k]
        ]

    def _enhance_query_with_history(self, query):
        if not self.conversation_history:
            return query
            
        # Add relevant context from conversation history
        recent_context = " ".join(self.conversation_history[-2:])
        return f"{query} Context: {recent_context}"

    def clear_cache(self):
        """Clear the retrieval cache."""
        self._cached_retrieve.cache_clear()

    def update_weights(self, new_weights):
        """Update retrieval weights dynamically."""
        if set(new_weights.keys()) == set(self.weights.keys()):
            total = sum(new_weights.values())
            self.weights = {k: v/total for k, v in new_weights.items()}
            self.clear_cache()  # Clear cache as weights changed
