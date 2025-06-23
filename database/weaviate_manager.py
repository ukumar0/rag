import weaviate
import weaviate.classes as wvc
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import List, Dict, Any, Optional
import logging
from config import OPENAI_API_KEY
import os
import json
import numpy as np

logger = logging.getLogger(__name__)

class WeaviateManager:
    def __init__(self, 
                 weaviate_url: str = "https://your-cluster-url.weaviate.network",
                 weaviate_api_key: str = None,
                 index_name: str = "RagDocuments"):
        """
        Initialize Weaviate manager with LlamaIndex integration.
        
        For local development without Docker, you can use:
        1. Weaviate Cloud Services (WCS) free tier
        2. A remote Weaviate instance
        
        Args:
            weaviate_url: URL of your Weaviate instance
            weaviate_api_key: API key for authentication (if required)
            index_name: Name of the collection/index in Weaviate
        """
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.index_name = index_name
        self.client = None
        self.vector_store = None
        self.index = None
        self.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
        
    def connect(self):
        """Connect to Weaviate instance"""
        try:
            # Check if this is a local connection
            if "localhost" in self.weaviate_url or "127.0.0.1" in self.weaviate_url:
                # Local Weaviate connection
                host = self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
                port = 8080
                if ":" in self.weaviate_url and self.weaviate_url.count(":") >= 2:
                    port = int(self.weaviate_url.split(":")[-1])
                
                self.client = weaviate.connect_to_local(host=host, port=port)
                
            elif self.weaviate_api_key:
                # Authenticated cloud connection
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.weaviate_url,
                    auth_credentials=wvc.init.Auth.api_key(self.weaviate_api_key),
                )
            else:
                # No API key provided - cannot connect to cloud instances
                logger.error("Cannot connect to cloud Weaviate instance without API key")
                logger.info("Either provide a weaviate_api_key or use a local Weaviate instance")
                logger.info("To set up Weaviate:")
                logger.info("1. Local: Use Docker - docker run -p 8080:8080 weaviate/weaviate:latest")
                logger.info("2. Cloud: Get free account at https://console.weaviate.cloud/")
                return False
            
            logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            
            # Initialize vector store with correct text field mapping
            self.vector_store = WeaviateVectorStore(
                weaviate_client=self.client,
                index_name=self.index_name,
                text_key="content"  # Map to our 'content' field instead of default 'text'
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize or load index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_collection_if_not_exists(self):
        """Create collection schema if it doesn't exist"""
        try:
            if not self.client.collections.exists(self.index_name):
                # Create collection with custom properties
                # Note: Using TEXT for metadata to avoid complex nested object schema
                self.client.collections.create(
                    name=self.index_name,
                    properties=[
                        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="filename", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                        wvc.config.Property(name="doc_hash", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="metadata_json", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="file_type", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="file_size", data_type=wvc.config.DataType.INT),
                    ],
                    # Configure vectorizer (we'll handle embeddings ourselves)
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                )
                logger.info(f"Created collection: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Store documents and their chunks in Weaviate using LlamaIndex integration
        """
        logger.info("Using LlamaIndex for document storage")
        return self.store_documents_with_llamaindex(documents)
    
    def store_documents_with_llamaindex(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Store documents using LlamaIndex with proper schema mapping
        
        This method uses LlamaIndex's capabilities while ensuring proper type handling.
        
        Note: We use string doc_id to avoid LlamaIndex's known issue where it converts 
        integer doc_id to string 'None'. Other integer metadata fields are still 
        monitored and may trigger fallback to direct Weaviate insertion if needed.
        """
        try:
            if not self.index:
                logger.error("LlamaIndex not initialized")
                return False
            
            # Check if we have integer metadata that will be corrupted by LlamaIndex
            # Note: We now use string doc_id to avoid LlamaIndex issues, but still check for other integer metadata
            has_problematic_integer_metadata = False
            for doc in documents:
                chunks = doc.get('chunks', [])
                for chunk in chunks:
                    metadata = chunk.get('metadata', {})
                    # Check for integer values in metadata (excluding file_size and chunk_index which we handle properly)
                    problematic_integers = {k: v for k, v in metadata.items() 
                                          if isinstance(v, int) and k not in ['file_size']}
                    if problematic_integers:
                        logger.debug(f"Found problematic integer metadata: {problematic_integers}")
                        has_problematic_integer_metadata = True
                        break
                if has_problematic_integer_metadata:
                    break
            
            if has_problematic_integer_metadata:
                logger.warning("  Detected problematic integer metadata fields. LlamaIndex has known issues with integer metadata:")
                logger.warning("    - Converts integers to floats automatically")
                logger.warning("    - May ignore actual metadata values and use hardcoded defaults")
                logger.info(" Using direct Weaviate insertion to preserve data integrity.")
                return self._direct_weaviate_insertion(documents)
            
            nodes = []
            valid_docs = []
            
            # First, validate all documents before processing
            for doc in documents:
                doc_id = doc.get('id')
                
                # Validate doc_id more thoroughly and convert to string for LlamaIndex compatibility
                if doc_id is None or doc_id == 'None' or doc_id == '':
                    logger.warning(f"Invalid doc_id for document: {doc_id}. Document: {doc.get('filename', 'unknown')}. Skipping.")
                    continue
                    
                try:
                    # Convert to integer first to validate it's a valid number
                    doc_id_int = int(doc_id)
                    if doc_id_int <= 0:
                        logger.warning(f"Invalid doc_id value {doc_id_int} for document: {doc.get('filename', 'unknown')}. Skipping.")
                        continue
                    # Convert to string for LlamaIndex compatibility
                    doc_id_str = str(doc_id_int)
                except (ValueError, TypeError):
                    logger.error(f"Cannot convert doc_id to integer: {doc_id} (type: {type(doc_id)}). Document: {doc.get('filename', 'unknown')}. Skipping document.")
                    continue
                
                # Add to valid documents list with string doc_id
                doc['id'] = doc_id_str  # Store as string for LlamaIndex compatibility
                valid_docs.append(doc)
            
            if not valid_docs:
                logger.error("No valid documents to process after validation")
                return False
            
            logger.info(f"Processing {len(valid_docs)} valid documents out of {len(documents)} total")
            
            # Process valid documents
            for doc in valid_docs:
                doc_id = doc['id']  # Already validated and converted to string
                
                doc_hash = doc.get('hash', '')
                filename = doc.get('filename', 'unknown')
                
                # Process chunks
                chunks = doc.get('chunks', [])
                for idx, chunk in enumerate(chunks):
                    import json
                    original_metadata = chunk.get('metadata', {})
                    
                    # Create properly typed metadata for LlamaIndex
                    # Using string doc_id to avoid LlamaIndex conversion issues
                    metadata = {
                        'filename': str(filename),
                        'doc_id': str(doc_id),  # Use string to avoid LlamaIndex issues
                        'chunk_index': int(idx),  # Keep as integer for Weaviate schema
                        'doc_hash': str(doc_hash),
                        'metadata_json': json.dumps(original_metadata),
                        'file_type': str(original_metadata.get('file_type', 'unknown')),
                        'file_size': int(original_metadata.get('file_size', 0))  # Keep as integer for Weaviate schema
                    }
                    
                    # Generate a proper UUID for the node
                    import uuid
                    node_uuid = str(uuid.uuid4())
                    
                    node = TextNode(
                        text=chunk['content'],
                        metadata=metadata,
                        id_=node_uuid  # Use proper UUID
                    )
                    
                    nodes.append(node)
            
            if nodes:
                # Use LlamaIndex's insert_nodes with our validated data
                try:
                    # Get the count before insertion
                    initial_count = self._get_weaviate_object_count()
                    
                    logger.info(f"Inserting {len(nodes)} nodes via LlamaIndex...")
                    
                    self.index.insert_nodes(nodes)
                    
                    # Check if insertion was actually successful
                    final_count = self._get_weaviate_object_count()
                    objects_added = final_count - initial_count
                    
                    if objects_added > 0:
                        logger.info(f"Successfully stored {objects_added} chunks using LlamaIndex")
                        return True
                    else:
                        logger.warning("LlamaIndex insertion completed but no objects were actually added to Weaviate")
                        # Fallback: try direct Weaviate insertion
                        logger.info("Attempting direct Weaviate insertion as fallback...")
                        return self._direct_weaviate_insertion(valid_docs)
                        
                except Exception as e:
                    logger.error(f"LlamaIndex insertion failed with exception: {str(e)}")
                    # Fallback: try direct Weaviate insertion
                    logger.info("Attempting direct Weaviate insertion as fallback...")
                    return self._direct_weaviate_insertion(valid_docs)
            else:
                logger.warning("No valid nodes to store")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store documents with LlamaIndex: {e}")
            return False
    
    def _direct_weaviate_insertion(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Direct Weaviate insertion bypassing LlamaIndex for problematic cases
        """
        try:
            if not self.client:
                logger.error("Weaviate client not available for direct insertion")
                return False
            
            collection = self.client.collections.get(self.index_name)
            batch_data = []
            
            for doc in documents:
                doc_id = doc['id']  # Already validated and converted to string
                doc_hash = doc.get('hash', '')
                filename = doc.get('filename', 'unknown')
                
                chunks = doc.get('chunks', [])
                for idx, chunk in enumerate(chunks):
                    original_metadata = chunk.get('metadata', {})
                    
                    # Create the data object with explicit type casting
                    data_object = {
                        'content': str(chunk['content']),
                        'filename': str(filename),
                        'doc_id': str(doc_id),  # Use string for consistency with LlamaIndex approach
                        'chunk_index': int(idx),  # Explicit integer conversion
                        'doc_hash': str(doc_hash),
                        'metadata_json': json.dumps(original_metadata),
                        'file_type': str(original_metadata.get('file_type', 'unknown')),
                        'file_size': int(original_metadata.get('file_size', 0))  # Explicit integer conversion
                    }
                    
                    # Get embedding from chunk
                    if 'embedding' in chunk:
                        embedding = chunk['embedding']
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        elif isinstance(embedding, str):
                            # Handle JSON string embeddings
                            try:
                                embedding = json.loads(embedding)
                            except:
                                logger.warning(f"Could not parse embedding for chunk {idx}")
                                embedding = None
                    else:
                        # Generate embedding if not present
                        embedding = self.embed_model.get_text_embedding(chunk['content'])
                    
                    # For Weaviate client v4, use the DataObject format
                    from weaviate.classes.data import DataObject
                    
                    batch_data.append(DataObject(
                        properties=data_object,
                        vector=embedding
                    ))
            
            if batch_data:
                # Insert in batches
                batch_size = 50
                for i in range(0, len(batch_data), batch_size):
                    batch = batch_data[i:i + batch_size]
                    try:
                        collection.data.insert_many(batch)
                        logger.info(f"Direct Weaviate insertion: batch {i//batch_size + 1} completed ({len(batch)} objects)")
                    except Exception as e:
                        logger.error(f"Direct Weaviate insertion failed for batch {i//batch_size + 1}: {str(e)}")
                        return False
                
                logger.info(f"Successfully inserted {len(batch_data)} objects via direct Weaviate insertion")
                return True
            else:
                logger.warning("No data to insert via direct method")
                return False
                
        except Exception as e:
            logger.error(f"Direct Weaviate insertion failed: {str(e)}")
            return False
    
    def _get_weaviate_object_count(self) -> int:
        """Get the current count of objects in Weaviate collection"""
        try:
            if not self.client:
                return 0
                
            collection = self.client.collections.get(self.index_name)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count if result.total_count else 0
        except Exception as e:
            logger.warning(f"Could not get Weaviate object count: {str(e)}")
            return 0
    

    
    def search_similar(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity with LlamaIndex integration
        """
        return self.search_similar_with_llamaindex(query, top_k, similarity_threshold)
    
    def search_similar_with_llamaindex(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using LlamaIndex retriever
        
        This method uses LlamaIndex's retriever interface
        """
        try:
            if not self.index:
                raise ValueError("LlamaIndex not properly initialized")
            
            logger.info(f"\n LLAMAINDEX VECTOR SEARCH:")
            logger.info(f"{'─'*50}")
            logger.info(f"Query: {query}")
            logger.info(f"Top-K: {top_k}")
            logger.info(f"Similarity Threshold: {similarity_threshold}")
            logger.info(f"{'─'*50}")
            
            # Force pure vector search instead of hybrid to get proper scores
            from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
            
            # Get query embedding manually
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Create vector store query with pure vector mode
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                mode=VectorStoreQueryMode.DEFAULT  # Use pure vector search
            )
            
            logger.info(f" PERFORMING PURE VECTOR SEARCH...")
            
            # Query the vector store directly
            vector_store_result = self.vector_store.query(vector_query)
            
            logger.info(f" RETRIEVED {len(vector_store_result.nodes)} NODES FROM VECTOR STORE")
            
            results = []
            for i, (node, similarity) in enumerate(zip(vector_store_result.nodes, vector_store_result.similarities), 1):
                # Use the similarity score directly from vector store
                score = similarity if similarity is not None else 0.0
                
                logger.info(f"\n Node {i}:")
                logger.info(f"    Score: {score:.4f}")
                logger.info(f"    Content Preview: {node.text[:200].replace(chr(10), ' ')}...")
                if hasattr(node, 'metadata') and 'filename' in node.metadata:
                    logger.info(f"    Source: {node.metadata['filename']}")
                
                if score >= similarity_threshold:
                    # Metadata should already be properly typed as integers
                    metadata = node.metadata.copy()
                    
                    result = {
                        'content': node.text,
                        'metadata': metadata,
                        'score': float(score),
                        'similarity': float(score),
                        'node_id': node.node_id if hasattr(node, 'node_id') else None
                    }
                    results.append(result)
                else:
                    logger.info(f"     Filtered out (score {score:.4f} < threshold {similarity_threshold})")
            
            logger.info(f"\n LLAMAINDEX SEARCH RESULTS: {len(results)} documents above threshold")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search with LlamaIndex: {e}")
            logger.exception(e)
            return []
    

    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Weaviate collection"""
        try:
            if not self.client:
                return {"error": "Not connected"}
            
            collection = self.client.collections.get(self.index_name)
            response = collection.aggregate.over_all(total_count=True)
            
            return {
                "total_objects": response.total_count,
                "collection_name": self.index_name,
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        try:
            if self.client.collections.exists(self.index_name):
                self.client.collections.delete(self.index_name)
                logger.info(f"Deleted collection: {self.index_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def close(self):
        """Close the Weaviate connection"""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection") 