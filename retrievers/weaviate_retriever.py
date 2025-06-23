from typing import List, Dict, Any
from database.weaviate_manager import WeaviateManager
from config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_INDEX_NAME
import logging

logger = logging.getLogger(__name__)

class WeaviateRetriever:
    def __init__(self, 
                 weaviate_url: str = None,
                 weaviate_api_key: str = None,
                 index_name: str = None):
        """
        Initialize Weaviate retriever
        
        Args:
            weaviate_url: Weaviate instance URL (defaults to config)
            weaviate_api_key: API key for authentication (defaults to config)
            index_name: Collection name (defaults to config)
        """
        self.weaviate_manager = WeaviateManager(
            weaviate_url=weaviate_url or WEAVIATE_URL,
            weaviate_api_key=weaviate_api_key or WEAVIATE_API_KEY,
            index_name=index_name or WEAVIATE_INDEX_NAME
        )
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Weaviate and initialize"""
        if not self.is_connected:
            success = self.weaviate_manager.connect()
            if success:
                self.weaviate_manager.create_collection_if_not_exists()
                self.is_connected = True
            return success
        return True
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents with their metadata.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
                      or documents with 'chunks' containing chunked content
        """
        if not self.connect():
            raise RuntimeError("Failed to connect to Weaviate")
        
        # Convert documents to the format expected by WeaviateManager
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                if 'chunks' in doc:
                    # Document already has chunks
                    formatted_docs.append(doc)
                elif 'content' in doc:
                    # Single document, create a chunk
                    formatted_doc = {
                        'id': i,
                        'hash': f"doc_{i}",
                        'filename': doc.get('metadata', {}).get('filename', f'doc_{i}'),
                        'chunks': [{
                            'content': doc['content'],
                            'metadata': doc.get('metadata', {})
                        }]
                    }
                    formatted_docs.append(formatted_doc)
                else:
                    # Handle legacy format where doc is just text
                    formatted_doc = {
                        'id': i,
                        'hash': f"doc_{i}",
                        'filename': f'doc_{i}',
                        'chunks': [{
                            'content': str(doc),
                            'metadata': {}
                        }]
                    }
                    formatted_docs.append(formatted_doc)
            else:
                # Handle legacy format where doc is just text
                formatted_doc = {
                    'id': i,
                    'hash': f"doc_{i}",
                    'filename': f'doc_{i}',
                    'chunks': [{
                        'content': str(doc),
                        'metadata': {}
                    }]
                }
                formatted_docs.append(formatted_doc)
        
        # Store in Weaviate
        self.weaviate_manager.store_documents(formatted_docs)
        logger.info(f"Indexed {len(formatted_docs)} documents in Weaviate")

    def build_index(self, texts: List[str]) -> None:
        """Legacy method for backward compatibility"""
        documents = [{'content': text, 'metadata': {}} for text in texts]
        self.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries containing document content, metadata, and relevance score
        """
        if not self.connect():
            raise RuntimeError("Failed to connect to Weaviate")
        
        results = self.weaviate_manager.search_similar(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results for compatibility with existing retrievers
        formatted_results = []
        for result in results:
            formatted_result = {
                'content': result['content'],
                'metadata': result['metadata'],
                'score': result['score']
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Weaviate collection"""
        if not self.connect():
            return {"error": "Failed to connect to Weaviate"}
        
        return self.weaviate_manager.get_stats()
    
    def close(self):
        """Close the Weaviate connection"""
        if self.is_connected:
            self.weaviate_manager.close()
            self.is_connected = False 