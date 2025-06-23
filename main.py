import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

import argparse
import os
import tiktoken
from openai import OpenAI
from config import OPENAI_API_KEY, SIMILARITY_THRESHOLD
from query.query_transformer import QueryTransformer
from query.query_router import QueryRouter
from retrievers.dense_retriever import DenseRetriever
from retrievers.weaviate_retriever import WeaviateRetriever
from retrievers.sparse_retriever import SparseRetriever
from retrievers.multi_vector_retriever import MultiVectorRetriever
from retrievers.hybrid_retriever import HybridRetriever

from rerankers.multi_stage_reranker import MultiStageReranker
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.context_distiller import ContextDistiller
from generation.multi_step_reasoner import MultiStepReasoner
from generation.self_reflective_rag import SelfReflectiveRAG
from agents.enhanced_rag_agent import EnhancedRAGAgent
from database.db_manager import DatabaseManager
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from chunking.enhanced_chunker import EnhancedChunker
from metadata.metadata_enhancer import MetadataEnhancer
from utils.html_cleaner import HTMLCleaner
import hashlib
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI LLM Implementation
class OpenAILLM:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content

class AdvancedRAGPipeline:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_embedding_tokens = 8000  # Leave some buffer below 8192 limit
        
        # Initialize databases
        self.db_manager = DatabaseManager()
        self.db_manager.init_db()
        
        # Initialize Weaviate as primary vector storage
        from database.weaviate_manager import WeaviateManager
        import os
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY')

        
        # TEMPORARILY DISABLE WEAVIATE - Force SQLite usage for debugging
        # Comment out this line to re-enable Weaviate
        # weaviate_url = None
        
        self.weaviate_manager = WeaviateManager(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key
        )
        
        # Try to connect to Weaviate
        self.use_weaviate = False
        if weaviate_url:
            if self.weaviate_manager.connect():
                self.weaviate_manager.create_collection_if_not_exists()
                self.use_weaviate = True
                logger.info("Using Weaviate for vector storage")
            else:
                logger.warning("Failed to connect to Weaviate, falling back to SQLite")
        else:
            logger.info("Weaviate not configured, using SQLite for vector storage")
        
        # Initialize LLM
        self.llm = OpenAILLM(model=model_name)
        
        # Initialize components
        self.chunker = EnhancedChunker()
        self.metadata_enhancer = MetadataEnhancer()
        self.html_cleaner = HTMLCleaner()
        
        # Initialize retrievers
        self.dense_retriever = DenseRetriever()  # Keep for fallback and batch embedding
        self.weaviate_retriever = WeaviateRetriever()  # Primary vector retriever
        self.sparse_retriever = SparseRetriever()
        self.multi_vector_retriever = MultiVectorRetriever()
        
        # Initialize hybrid retriever - using Weaviate as primary dense retriever for better performance
        self.hybrid_retriever = HybridRetriever(
            dense=self.weaviate_retriever,  # Use Weaviate for superior vector search
            sparse=self.sparse_retriever,
            multi=self.multi_vector_retriever
        )
        
        # Initialize rerankers
        self.reranker = MultiStageReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()
        
        # Initialize generation components
        self.context_distiller = ContextDistiller(self.llm)
        self.query_transformer = QueryTransformer(self.llm)
        self.query_router = QueryRouter()
        
        self.multi_step_reasoner = MultiStepReasoner(
            self.llm, 
            self.hybrid_retriever,
            db_manager=self.db_manager,
            dense_retriever=self.dense_retriever
        )
        
        reflection_template = """Query: "{query}"\nContext: "{context}"\nShould you retrieve? RETRIEVE or SKIP."""
        self.self_reflective_rag = SelfReflectiveRAG(
            self.llm,
            self.hybrid_retriever,
            reflection_template
        )
        
        # Initialize the Enhanced RAG Agent for intelligent orchestration
        self.rag_agent = EnhancedRAGAgent(self)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def _batch_create_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Create embeddings in batches for efficiency.
        
        Args:
            texts: List of text chunks to embed
            batch_size: Size of each batch (OpenAI supports up to 2048)
            
        Returns:
            List of embeddings corresponding to input texts
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.dense_retriever._get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error in batch embedding {i//batch_size + 1}: {str(e)}")
                # Fallback to individual processing for this batch
                for text in batch:
                    try:
                        embedding = self.dense_retriever._get_embeddings([text])[0]
                        all_embeddings.append(embedding)
                    except Exception as individual_error:
                        logger.error(f"Error creating individual embedding: {str(individual_error)}")
                        # Use zero vector as fallback
                        all_embeddings.append(np.zeros(1536))  # Default OpenAI embedding size
                        
        return all_embeddings

    def _fast_token_estimate(self, text: str) -> int:
        """
        Fast token estimation without expensive tiktoken encoding.
        Uses character-based approximation (1 token ≈ 4 characters for English).
        """
        return len(text) // 4

    def _simple_chunk_split(self, text: str, max_chars: int = 30000) -> List[str]:
        """
        Simple and fast text splitting by sentences and paragraphs.
        Much faster than token-based splitting for large documents.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_chars:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk + sentence) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                        else:
                            # Even single sentence is too long, split by words
                            words = sentence.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk + word) > max_chars:
                                    if word_chunk:
                                        chunks.append(word_chunk.strip())
                                        word_chunk = word + " "
                                    else:
                                        # Single word too long, truncate
                                        chunks.append(word[:max_chars])
                                else:
                                    word_chunk += word + " "
                            if word_chunk:
                                current_chunk = word_chunk
                    else:
                        current_chunk += sentence + ". "
            else:
                # Regular paragraph handling
                if len(current_chunk + paragraph) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph + "\n\n"
                    else:
                        chunks.append(paragraph)
                else:
                    current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _split_large_chunk(self, chunk: str, max_tokens: int = None) -> List[str]:
        """Split chunks that are too large for embedding model"""
        if max_tokens is None:
            max_tokens = self.max_embedding_tokens
            
        token_count = self._count_tokens(chunk)
        if token_count <= max_tokens:
            return [chunk]
        
        # Split by sentences first
        sentences = chunk.split('. ')
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + ". "
            if self._count_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    # Even single sentence is too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + word + " "
                        if self._count_tokens(test_word_chunk) > max_tokens:
                            if word_chunk:
                                sub_chunks.append(word_chunk.strip())
                                word_chunk = word + " "
                            else:
                                # Single word too long (very rare), truncate
                                sub_chunks.append(word[:max_tokens//2])
                        else:
                            word_chunk = test_word_chunk
                    if word_chunk:
                        current_chunk = word_chunk
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
            
        return sub_chunks

    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for context"""
        if not conversation_history:
            return ""
        
        formatted = "Previous conversation context:\n"
        for i, turn in enumerate(conversation_history[-3:], 1):  # Last 3 turns
            formatted += f"{i}. User: {turn['query']}\n"
            formatted += f"   Assistant: {turn['response'][:200]}{'...' if len(turn['response']) > 200 else ''}\n\n"
        
        return formatted

    def _get_document_hash(self, doc: Dict[str, str]) -> str:
        """Generate a unique hash for a document based on its content and metadata."""
        content = doc.get('content', '')
        metadata = str(doc.get('metadata', {}))
        return hashlib.md5((content + metadata).encode()).hexdigest()

    def reset_database(self):
        """Reset the database to handle embedding dimension mismatches"""
        logger.info("Resetting database to clear incompatible embeddings...")
        self.db_manager.clear_all_data()
        logger.info("Database reset complete")
    
    def reset_weaviate_database(self):
        """Reset the Weaviate database by deleting the collection"""
        try:
            from database.weaviate_manager import WeaviateManager
            import os
            
            # Check if Weaviate credentials are configured
            weaviate_url = os.getenv('WEAVIATE_URL')
            weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
            
            if not weaviate_url:
                logger.warning("Weaviate is not configured (no WEAVIATE_URL found)")
                logger.info("To use Weaviate, set up environment variables:")
                logger.info("1. For local: WEAVIATE_URL=http://localhost:8080")
                logger.info("2. For cloud: WEAVIATE_URL=https://your-cluster.weaviate.network and WEAVIATE_API_KEY=your-key")
                logger.info("Currently using local SQLite database only.")
                return
            
            # Create WeaviateManager with environment variables
            weaviate_manager = WeaviateManager(
                weaviate_url=weaviate_url,
                weaviate_api_key=weaviate_api_key
            )
            
            if weaviate_manager.connect():
                result = weaviate_manager.delete_collection()
                if result:
                    logger.info("Weaviate collection deleted successfully")
                    # Recreate the collection
                    weaviate_manager.create_collection_if_not_exists()
                    logger.info("Weaviate collection recreated")
                else:
                    logger.warning("Weaviate collection was not found or already empty")
                weaviate_manager.close()
            else:
                logger.error("Could not connect to Weaviate. Please check your configuration.")
                logger.info("Make sure Weaviate is running and credentials are correct.")
        except Exception as e:
            logger.error(f"Error resetting Weaviate database: {str(e)}")
            logger.info("If you're not using Weaviate, you can ignore this error and use --reset_db instead.")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about current storage configuration"""
        info = {
            'primary_storage': 'Weaviate' if self.use_weaviate else 'SQLite',
            'weaviate_available': self.use_weaviate,
            'sqlite_available': True,
        }
        
        # Get SQLite stats
        sqlite_stats = self.db_manager.get_stats()
        info['sqlite_stats'] = sqlite_stats
        
        # Get Weaviate stats if available
        if self.use_weaviate:
            try:
                weaviate_stats = self.weaviate_manager.get_stats()
                info['weaviate_stats'] = weaviate_stats
            except Exception as e:
                info['weaviate_stats'] = {'error': str(e)}
        
        return info
    
    def close_connections(self):
        """Close all database connections properly"""
        if self.use_weaviate and hasattr(self.weaviate_manager, 'close'):
            try:
                self.weaviate_manager.close()
                logger.debug("Weaviate connection closed")
            except Exception as e:
                logger.debug(f"Error closing Weaviate connection: {str(e)}")

    def _batch_create_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Create embeddings in batches for efficiency.
        
        Args:
            texts: List of text chunks to embed
            batch_size: Size of each batch (OpenAI supports up to 2048)
            
        Returns:
            List of embeddings corresponding to input texts
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.dense_retriever._get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error in batch embedding {i//batch_size + 1}: {str(e)}")
                # Fallback to individual processing for this batch
                for text in batch:
                    try:
                        embedding = self.dense_retriever._get_embeddings([text])[0]
                        all_embeddings.append(embedding)
                    except Exception as individual_error:
                        logger.error(f"Error creating individual embedding: {str(individual_error)}")
                        # Use zero vector as fallback
                        all_embeddings.append(np.zeros(1536))  # Default OpenAI embedding size
                        
        return all_embeddings

    def _simple_chunk_split(self, text: str, max_chars: int = 30000) -> List[str]:
        """
        Simple and fast text splitting by sentences and paragraphs.
        Much faster than token-based splitting for large documents.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_chars:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk + sentence) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                        else:
                            # Even single sentence is too long, split by words
                            words = sentence.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk + word) > max_chars:
                                    if word_chunk:
                                        chunks.append(word_chunk.strip())
                                        word_chunk = word + " "
                                    else:
                                        # Single word too long, truncate
                                        chunks.append(word[:max_chars])
                                else:
                                    word_chunk += word + " "
                            if word_chunk:
                                current_chunk = word_chunk
                    else:
                        current_chunk += sentence + ". "
            else:
                # Regular paragraph handling
                if len(current_chunk + paragraph) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph + "\n\n"
                    else:
                        chunks.append(paragraph)
                else:
                    current_chunk += paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def process_documents_optimized(self, documents: List[Dict[str, str]], chunking_strategy: str = 'fast') -> Dict[str, int]:
        """
        Optimized document processing with batch embedding generation.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            chunking_strategy: Strategy for chunking ('fast', 'semantic', 'hierarchical', 'hybrid')
            
        Returns:
            Dict containing processing statistics
        """
        logger.info(f"Processing {len(documents)} documents with optimized pipeline...")
        
        stats = {
            'total_documents': len(documents),
            'total_chunks': 0,
            'processed_documents': 0,
            'skipped_documents': 0,
            'failed_documents': 0,
            'oversized_chunks_split': 0
        }
        
        # Step 1: Collect all chunks from all documents first
        all_chunks_data = []
        document_chunk_mapping = []
        
        for doc in documents:
            doc_hash = self._get_document_hash(doc)
            
            # Skip if document already exists
            if self.db_manager.document_exists(doc_hash):
                stats['skipped_documents'] += 1
                logger.info(f"Skipping already processed document: {doc.get('metadata', {}).get('filename', 'unknown')}")
                continue
                
            try:
                # Clean HTML content if present
                content = doc['content']
                metadata = doc.get('metadata', {})
                
                if self.html_cleaner.is_html_content(content):
                    html_result = self.html_cleaner.clean_and_extract_metadata(content)
                    content = html_result['cleaned_text']
                    metadata.update({
                        'had_html_content': True,
                        'original_text_length': len(doc['content']),
                        'cleaned_text_length': html_result['text_length'],
                    })
                else:
                    metadata['had_html_content'] = False
                
                # Store document
                stored_doc = self.db_manager.store_document(
                    content=content,
                    metadata=metadata,
                    doc_hash=doc_hash
                )
                
                # Validate document storage
                if not stored_doc or not hasattr(stored_doc, 'id') or stored_doc.id is None:
                    logger.error(f"Failed to store document or get valid ID: {metadata.get('filename', 'unknown')}")
                    stats['failed_documents'] += 1
                    continue
                
                logger.debug(f"Document stored with ID: {stored_doc.id}")
                
                # Generate chunks based on strategy
                if chunking_strategy == 'fast':
                    chunks = self._simple_chunk_split(content, max_chars=30000)  # ~7500 tokens
                elif chunking_strategy == 'semantic':
                    chunks = self.chunker._semantic_chunking(content)
                elif chunking_strategy == 'hierarchical':
                    chunks = self.chunker._hierarchical_chunking(content)
                else:
                    chunks = self.chunker.process_with_strategy(content, strategy=chunking_strategy, clean_html=False)
                
                # Prepare chunk data for batch processing
                chunk_position = 0  # Track position within this document
                for chunk_idx, chunk in enumerate(chunks):
                    # Fast token estimation instead of expensive counting
                    estimated_tokens = self._fast_token_estimate(chunk)
                    
                    if estimated_tokens > 30000:  # Split very large chunks
                        sub_chunks = self._simple_chunk_split(chunk, max_chars=30000)
                        stats['oversized_chunks_split'] += 1
                    else:
                        sub_chunks = [chunk]
                    
                    for sub_chunk in sub_chunks:
                        # Enhance metadata (but don't create embeddings yet)
                        enhanced_metadata = self.metadata_enhancer.enhance(
                            text=sub_chunk,
                            base_metadata=metadata
                        )
                        
                        chunk_data = {
                            'content': sub_chunk,
                            'metadata': enhanced_metadata,
                            'doc_id': stored_doc.id,
                            'doc_hash': doc_hash,
                            'doc_metadata': metadata,
                            'chunk_index': chunk_position  # Position within this specific document
                        }
                        
                        all_chunks_data.append(chunk_data)
                        document_chunk_mapping.append({
                            'doc_id': stored_doc.id,
                            'chunk_data': chunk_data
                        })
                        chunk_position += 1  # Increment for next chunk in this document
                
                stats['processed_documents'] += 1
                logger.info(f"Prepared chunks for document: {metadata.get('filename', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to process document: {str(e)}")
                stats['failed_documents'] += 1
        
        # Step 2: Batch generate embeddings for ALL chunks at once
        if all_chunks_data:
            logger.info(f"Generating embeddings for {len(all_chunks_data)} chunks in batches...")
            
            chunk_texts = [chunk_data['content'] for chunk_data in all_chunks_data]
            embeddings = self._batch_create_embeddings(chunk_texts, batch_size=100)
            
            # Step 3: Store all chunks with their embeddings
            logger.info("Storing chunks in database...")
            
            # Group chunks by document for efficient storage
            doc_chunks = {}
            for i, chunk_data in enumerate(all_chunks_data):
                doc_id = chunk_data['doc_id']
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                
                enhanced_chunk = {
                    'content': chunk_data['content'],
                    'metadata': chunk_data['metadata'],
                    'embedding': embeddings[i] if i < len(embeddings) else np.zeros(1536)
                }
                doc_chunks[doc_id].append(enhanced_chunk)
            
            # Store chunks grouped by document
            for doc_id, chunks in doc_chunks.items():
                try:
                    # Store in SQLite
                    self.db_manager.store_chunks(doc_id, chunks)
                    
                    # Store in Weaviate if available
                    if self.use_weaviate:
                        # Find document metadata for this doc_id
                        doc_metadata = None
                        doc_hash = None
                        for chunk_mapping in document_chunk_mapping:
                            if chunk_mapping['doc_id'] == doc_id:
                                doc_metadata = chunk_mapping['chunk_data']['doc_metadata']
                                doc_hash = chunk_mapping['chunk_data']['doc_hash']
                                break
                        
                        if doc_metadata:
                            weaviate_doc = {
                                'id': doc_id,
                                'hash': doc_hash,
                                'filename': doc_metadata.get('filename', 'unknown'),
                                'chunks': chunks
                            }
                            success = self.weaviate_manager.store_documents([weaviate_doc])
                            if success:
                                logger.debug(f"Stored {len(chunks)} chunks in Weaviate for doc {doc_id}")
                        else:
                            logger.warning(f"No metadata found for doc_id {doc_id}, skipping Weaviate storage")
                    
                    stats['total_chunks'] += len(chunks)
                    
                except Exception as e:
                    logger.error(f"Error storing chunks for document {doc_id}: {str(e)}")
        
        logger.info("Optimized processing complete. Statistics:")
        logger.info(f"- Total documents: {stats['total_documents']}")
        logger.info(f"- Successfully processed: {stats['processed_documents']}")
        logger.info(f"- Skipped (already processed): {stats['skipped_documents']}")
        logger.info(f"- Failed documents: {stats['failed_documents']}")
        logger.info(f"- Total chunks generated: {stats['total_chunks']}")
        logger.info(f"- Oversized chunks split: {stats['oversized_chunks_split']}")
        
        return stats

    def query(self, 
             user_query: str, 
             conversation_history: Optional[List[Dict]] = None,
             top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user query and return relevant information.
        
        Args:
            user_query: The user's question or query
            conversation_history: Optional list of previous conversation turns
            top_k: Number of top results to retrieve
            
        Returns:
            Dict containing response, sources, and reasoning steps
        """
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Format conversation history for context
            conversation_context = self._format_conversation_history(conversation_history)
            
            # Create enhanced query with conversation context
            if conversation_context and any(keyword in user_query.lower() 
                                          for keyword in ['discuss', 'talked', 'mentioned', 'said', 'chat', 'conversation', 'before']):
                enhanced_query = f"{user_query}\n\nConversation context:\n{conversation_context}"
            else:
                enhanced_query = user_query
            
            # Transform query
            transformed_query = self.query_transformer.transform_query(
                enhanced_query,
                strategy="auto"
            )
            
            # Validate transformed query
            if not transformed_query or not transformed_query.strip():
                transformed_query = user_query.strip()
                if not transformed_query:
                    logger.warning("Both original and transformed queries are empty")
                    return {
                        "query": user_query,
                        "response": "I apologize, but I didn't receive a valid query to process.",
                        "error": "Empty query"
                    }
            
            # Retrieve relevant chunks from vector database
            if self.use_weaviate:
                # Use Weaviate for vector search
                try:
                    # Use existing weaviate manager
                    wm = self.weaviate_manager
                    
                    if wm.connect():
                        print("Connected to Weaviate successfully")
                        
                        # Try with a more lenient similarity threshold first
                        logger.info(" Attempting retrieval with lenient threshold (0.3)")
                        initial_chunks = wm.search_similar(query=transformed_query, top_k=10, similarity_threshold=0.3)
                        
                        if initial_chunks:
                            print(f"Found {len(initial_chunks)} chunks with lenient threshold")
                            
                            # Now filter to best results
                            retrieved_chunks = [chunk for chunk in initial_chunks if chunk.get('score', 0) >= 0.5]
                            if not retrieved_chunks:
                                retrieved_chunks = initial_chunks[:5]  # Take top 5 if none meet stricter threshold
                                
                        else:
                            # Fallback: try with even more lenient threshold
                            logger.info("Trying with very lenient threshold (0.1)")
                            retrieved_chunks = wm.search_similar(query=transformed_query, top_k=10, similarity_threshold=0.1)
                            
                        if not retrieved_chunks:
                            logger.warning(" No chunks found even with lenient thresholds")
                            print("No relevant documents found. This might be due to:")
                            print("1. Empty or corrupted database")
                            print("2. Embedding model mismatch")
                            print("3. LlamaIndex/Weaviate scoring issues")
                            # Fallback to SQLite
                            query_embedding = self.dense_retriever._get_embeddings([transformed_query])[0]
                            retrieved_chunks = self.db_manager.search_similar_chunks(
                                query_embedding=query_embedding,
                                top_k=top_k
                            )
                            
                except Exception as e:
                    logger.error(f"Error retrieving from Weaviate: {str(e)}")
                    logger.info("Falling back to SQLite retrieval")
                    # Fallback to SQLite
                    query_embedding = self.dense_retriever._get_embeddings([transformed_query])[0]
                    retrieved_chunks = self.db_manager.search_similar_chunks(
                        query_embedding=query_embedding,
                        top_k=top_k
                    )
            else:
                # Use SQLite for vector search
                query_embedding = self.dense_retriever._get_embeddings([transformed_query])[0]
                retrieved_chunks = self.db_manager.search_similar_chunks(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            
            # DEBUG: Check retrieved chunks
            logger.debug(f"DEBUG: Retrieved {len(retrieved_chunks)} chunks")
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Show first 3
                content = chunk.get('content', '')
                logger.debug(f"DEBUG: Chunk {i}: content_len={len(content)}, preview='{content[:100]}'")
            
            # Rerank results using multi-stage reranker
            reranked_chunks = self.reranker.rerank(
                query=transformed_query,
                documents=retrieved_chunks
            )
            
            # Optional: Apply cross-encoder reranking for even better results
            if len(reranked_chunks) > 3:  # Only if we have enough chunks
                try:
                    reranked_chunks = self.cross_encoder_reranker.rerank(
                        query=transformed_query,
                        documents=reranked_chunks,
                        top_k=min(5, len(reranked_chunks))
                    )
                    logger.debug("Applied cross-encoder reranking")
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed, using multi-stage results: {e}")
                    # Continue with multi-stage results
            
            # DEBUG: Check reranked chunks
            logger.debug(f"DEBUG: Reranked to {len(reranked_chunks)} chunks")
            for i, chunk in enumerate(reranked_chunks[:3]):  # Show first 3
                content = chunk.get('content', '')
                logger.debug(f"DEBUG: Reranked chunk {i}: content_len={len(content)}, preview='{content[:100]}'")
            
            # Filter out empty chunks before context distillation
            valid_chunks = [chunk for chunk in reranked_chunks if chunk.get('content', '').strip()]
            if not valid_chunks:
                logger.warning("No valid chunks found after filtering empty content")
                return {
                    "query": user_query,
                    "response": "I apologize, but I couldn't find any relevant information in the knowledge base to answer your query.",
                    "error": "No valid chunks found"
                }
            
            # Distill context
            distilled_context = self.context_distiller.distill_context(
                documents=[chunk['content'] for chunk in valid_chunks],
                query=transformed_query
            )
            
            # Generate response with conversation context
            response_prompt = f"""Query: {user_query}

{conversation_context}

Retrieved Context:
{distilled_context}

Please provide a comprehensive response that takes into account both the conversation history (if relevant) and the retrieved context."""

            response = self.llm.generate(response_prompt)
            
            return {
                "query": user_query,
                "transformed_query": transformed_query,
                "retrieved_chunks": retrieved_chunks,
                "reranked_chunks": reranked_chunks,
                "distilled_context": distilled_context,
                "response": response,
                "conversation_history": conversation_history
                        }
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            # Fallback to simple generation
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "error": str(e)
            }

    def generate_advanced(self, query: str, **kwargs) -> str:
        """
        Generate a response using advanced RAG techniques including multi-step reasoning and self-reflection.
        
        Args:
            query: The user's query
            **kwargs: Additional arguments for generation
            
        Returns:
            str: The generated response with advanced reasoning
        """
        try:
            logger.info(f" Starting advanced generation for: {query}")
            
            # Step 1: Multi-step reasoning for complex queries
            logger.info(" Performing multi-step reasoning...")
            reasoning_result = self.multi_step_reasoner.reason_step_by_step(query=query)
            
            # Step 2: Self-reflective generation with iterative improvement
            logger.info(" Applying self-reflective RAG...")
            final_answer = self.self_reflective_rag.generate_with_reflection(
                query=query,
                max_iterations=kwargs.get('max_iterations', 2)
            )
            
            logger.info(" Advanced generation complete")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in advanced generation: {str(e)}")
            # Fallback to regular query method
            logger.info(" Falling back to standard generation...")
            result = self.query(query)
            return result.get('response', f"Error in generation: {str(e)}")

    def generate(self, query: str, **kwargs) -> str:
        """
        Generate a response for the given query using the RAG pipeline.
        
        Args:
            query: The user's query
            **kwargs: Additional arguments for generation
            
        Returns:
            str: The generated response
        """
        # Use advanced generation by default
        return self.generate_advanced(query, **kwargs)
    
    def process_with_agent(self, query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process query using the Enhanced RAG Agent for intelligent technique selection
        
        Args:
            query: User's query
            conversation_context: Optional conversation history
            
        Returns:
            Dict containing response and metadata about processing
        """
        return self.rag_agent.process_query(query, conversation_context)


def load_documents(directory: str) -> List[Dict]:
    """Load documents from a directory."""
    documents = []
    supported_extensions = {'.txt', '.html', '.htm', '.pdf'}
    
    for filename in os.listdir(directory):
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension in supported_extensions:
            file_path = os.path.join(directory, filename)
            try:
                if file_extension == '.pdf':
                    # Handle PDF files
                    content = extract_text_from_pdf(file_path)
                    file_type = 'pdf'
                else:
                    # Handle text and HTML files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_type = 'html' if file_extension in {'.html', '.htm'} else 'text'
                
                if content.strip():  # Only add documents with content
                    documents.append({
                        'content': content,
                        'metadata': {
                            'filename': filename,
                            'source': 'file',
                            'path': file_path,
                            'file_type': file_type,
                            'file_extension': file_extension,
                            'file_size': len(content)
                        }
                    })
                else:
                    logger.warning(f"No content extracted from {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading file {filename}: {str(e)}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file with robust error handling and fallbacks."""
    import signal
    import os
    from contextlib import contextmanager
    
    @contextmanager
    def timeout(duration):
        """Timeout context manager to prevent hanging on problematic PDFs"""
        import threading
        import time
        
        def timeout_handler():
            time.sleep(duration)
            raise TimeoutError(f"PDF processing timed out after {duration} seconds")
        
        # For Windows compatibility, we'll use a simpler approach
        # Just yield without actual timeout - the page-level error handling will catch issues
        try:
            yield
        finally:
            pass
    
    def try_pypdf_extraction(pdf_path: str) -> str:
        """Try extraction with pypdf library"""
        from pypdf import PdfReader
        
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages: {os.path.basename(pdf_path)}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Show progress every 10 pages
                    if page_num % 10 == 0:
                        logger.info(f"Processing page {page_num + 1}/{total_pages}...")
                    
                    page_text = page.extract_text()
                    if page_text and page_text.strip():  # Only add non-empty pages
                        text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                    
                except KeyError as e:
                    # Handle font-related KeyErrors specifically
                    logger.warning(f"Font issue on page {page_num + 1}, skipping: {str(e)}")
                    continue
                except Exception as page_error:
                    # Handle any other page-specific errors
                    error_type = type(page_error).__name__
                    logger.warning(f"{error_type} on page {page_num + 1}, skipping: {str(page_error)}")
                    continue
        
        return text.strip()
    
    def try_pdfplumber_extraction(pdf_path: str) -> str:
        """Fallback: Try extraction with pdfplumber if available"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                    except Exception as page_error:
                        logger.warning(f"pdfplumber: Could not extract text from page {page_num + 1}: {str(page_error)}")
                        continue
            return text.strip()
        except ImportError:
            logger.debug("pdfplumber not available for fallback")
            return ""
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    # Main extraction logic
    try:
        logger.info(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
        
        # Try pypdf first
        try:
            text = try_pypdf_extraction(pdf_path)
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters using pypdf")
                return text
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {str(e)}")
        
        # Fallback to pdfplumber if pypdf fails
        logger.info("Trying fallback extraction with pdfplumber...")
        text = try_pdfplumber_extraction(pdf_path)
        if text.strip():
            logger.info(f"Successfully extracted {len(text)} characters using pdfplumber")
            return text
        
        # If both fail, return empty but log the issue
        logger.error(f"Could not extract text from PDF: {os.path.basename(pdf_path)}")
        logger.info("Consider manually converting the PDF to text or using a different PDF file")
        return ""
            
    except ImportError as e:
        logger.error(f"PDF library not installed: {str(e)}")
        logger.info("Install with: pip install pypdf pdfplumber")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting text from PDF {pdf_path}: {str(e)}")
        return ""


def main():
    parser = argparse.ArgumentParser(description='Advanced RAG System')
    parser.add_argument('--docs_dir', type=str, help='Directory containing documents to process')
    parser.add_argument('--reset_db', action='store_true', help='Reset local database (clears all data)')
    parser.add_argument('--reset_weaviate', action='store_true', help='Reset Weaviate database (clears all data)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (default)')
    parser.add_argument('--query', type=str, help='Single query to process (non-interactive)')
    parser.add_argument('--chunking_strategy', type=str, default='hybrid', 
                       choices=['semantic', 'hierarchical', 'advanced_hierarchical', 'hybrid', 'sentence_window', 'fast'],
                       help='Chunking strategy to use (default: hybrid)')
    parser.add_argument('--optimized', action='store_true', 
                       help='Use optimized batch processing for faster document processing')
    parser.add_argument('--enhanced_agent', action='store_true',
                       help='Use Enhanced RAG Agent with intelligent query processing and multi-step reasoning')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    logger.info("Initializing Advanced RAG Pipeline...")
    pipeline = AdvancedRAGPipeline()
    
    # Reset database if requested
    if args.reset_db:
        pipeline.reset_database()
        logger.info("Local database has been reset. You can now process documents with new embeddings.")
        return
    
    # Reset Weaviate database if requested
    if args.reset_weaviate:
        pipeline.reset_weaviate_database()
        logger.info("Weaviate database has been reset. You can now process documents with new embeddings.")
        return
    
    # Show storage information
    storage_info = pipeline.get_storage_info()
    logger.info(f"Primary storage: {storage_info['primary_storage']}")
    if storage_info['weaviate_available']:
        logger.info(" Weaviate connected for vector storage")
    else:
        logger.info(" Using SQLite for vector storage")
    
    stats = storage_info['sqlite_stats']
    logger.info(f"Database contains {stats['documents']} documents and {stats['chunks']} chunks")
    
    # Show available chunking strategies
    available_strategies = pipeline.chunker.get_available_strategies()
    logger.info(f"Available chunking strategies: {', '.join(available_strategies)}")
    logger.info(f"Using chunking strategy: {args.chunking_strategy}")
    
    # Show processing mode
    if args.enhanced_agent:
        logger.info(" Enhanced RAG Agent mode enabled - using intelligent query processing")
    else:
        logger.info(" Standard processing mode - use --enhanced_agent for advanced features")
    
    # Process documents if directory is provided
    if args.docs_dir:
        if not os.path.exists(args.docs_dir):
            logger.error(f"Directory not found: {args.docs_dir}")
            return
            
        documents = load_documents(args.docs_dir)
        if documents:
            logger.info(f"Processing {len(documents)} documents from {args.docs_dir}")
            try:
                if args.optimized:
                    logger.info("Using OPTIMIZED batch processing for faster performance...")
                    processing_stats = pipeline.process_documents_optimized(documents, chunking_strategy=args.chunking_strategy)
                else:
                    logger.info("Using standard document processing...")
                    processing_stats = pipeline.process_documents_optimized(documents, chunking_strategy=args.chunking_strategy)
                logger.info("Document processing complete")
                print("\n" + "="*50)
                print("DOCUMENT PROCESSING SUMMARY")
                print("="*50)
                for key, value in processing_stats.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                print("="*50)
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                if "not aligned" in str(e):
                    logger.error("This might be due to embedding dimension mismatch.")
                    logger.error("Try running with --reset_db to clear old embeddings.")
                return
    
    # Handle single query
    if args.query:
        try:
            if args.enhanced_agent:
                logger.info(" Processing with Enhanced RAG Agent...")
                result = pipeline.process_with_agent(query=args.query)
                print(f"\nQuery: {args.query}")
                print(f"Transformed Query: {result.get('transformed_query', 'N/A')}")
                print(f"Retrieval Strategy: {result.get('retrieval_strategy', 'N/A')}")
                print(f"Documents Retrieved: {result.get('documents_retrieved', 0)}")
                print(f"Documents Used: {result.get('documents_used', 0)}")
                print(f"Confidence: {result.get('confidence', 0.0):.2f}")
                print(f"\nResponse: {result['answer']}")
                if result.get('reasoning_chain'):
                    print(f"\nReasoning Chain:")
                    for i, step in enumerate(result['reasoning_chain'], 1):
                        print(f"  {i}. {step}")
            else:
                result = pipeline.query(user_query=args.query)
                print(f"\nQuery: {args.query}")
                print(f"Response: {result['response']}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
        return
    
    # Interactive mode (default)
    if not args.query:
        print("\n" + "="*60)
        print(" ADVANCED RAG SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Commands:")
        print("  - Type your questions naturally")
        print("  - 'exit' or 'quit' to exit")
        print("  - 'history' to see conversation history")
        print("  - 'stats' to see database statistics")
        if args.enhanced_agent:
            print("  - 'mode standard' to switch to standard processing")
            print(" Enhanced RAG Agent Mode: Advanced query processing enabled")
        else:
            print("  - 'mode enhanced' to switch to enhanced agent processing")
        print("="*60)
        
        conversation_history = []
        
        while True:
            try:
                print("\n Enter your query: ", end='')
                query = input().strip()
                
                if query.lower() in ['exit', 'quit']:
                    print("\n Goodbye!")
                    break
                elif query.lower() == 'history':
                    print("\n Conversation History:")
                    if not conversation_history:
                        print("No conversation history yet.")
                    else:
                        for i, turn in enumerate(conversation_history, 1):
                            print(f"\n{i}. User: {turn['query']}")
                            print(f"   Assistant: {turn['response'][:200]}{'...' if len(turn['response']) > 200 else ''}")
                    continue
                elif query.lower() == 'stats':
                    storage_info = pipeline.get_storage_info()
                    print(f"\n  Storage Information:")
                    print(f"   Primary Storage: {storage_info['primary_storage']}")
                    print(f"   Weaviate Available: {storage_info['weaviate_available']}")
                    print(f"   Processing Mode: {'Enhanced RAG Agent' if args.enhanced_agent else 'Standard'}")
                    
                    print(f"\n  SQLite Statistics:")
                    sqlite_stats = storage_info['sqlite_stats']
                    print(f"   Documents: {sqlite_stats['documents']}")
                    print(f"   Chunks: {sqlite_stats['chunks']}")
                    
                    if storage_info['weaviate_available']:
                        print(f"\n  Weaviate Statistics:")
                        weaviate_stats = storage_info.get('weaviate_stats', {})
                        if 'error' in weaviate_stats:
                            print(f"   Error: {weaviate_stats['error']}")
                        else:
                            print(f"   Total Objects: {weaviate_stats.get('total_objects', 'N/A')}")
                            print(f"   Collection: {weaviate_stats.get('collection_name', 'N/A')}")
                    continue
                elif query.lower() == 'mode enhanced':
                    if not args.enhanced_agent:
                        args.enhanced_agent = True
                        print("\n Switched to Enhanced RAG Agent mode")
                        print("   Advanced query processing and multi-step reasoning enabled")
                    else:
                        print("\n  Already in Enhanced RAG Agent mode")
                    continue
                elif query.lower() == 'mode standard':
                    if args.enhanced_agent:
                        args.enhanced_agent = False
                        print("\n Switched to Standard processing mode")
                        print("   Use 'mode enhanced' to enable advanced features")
                    else:
                        print("\n  Already in Standard processing mode")
                    continue
                elif not query:
                    continue
                
                try:
                    print("\n Processing your query...")
                    
                    if args.enhanced_agent:
                        print(" Using Enhanced RAG Agent...")
                        result = pipeline.process_with_agent(
                            query=query,
                            conversation_context=conversation_history
                        )
                        
                        print(f"\n Query Analysis:")
                        print(f"   Original: {query}")
                        print(f"   Transformed: {result.get('transformed_query', 'N/A')}")
                        print(f"   Strategy: {result.get('retrieval_strategy', 'N/A')}")
                        print(f"   Documents Retrieved: {result.get('documents_retrieved', 0)}")
                        print(f"   Documents Used: {result.get('documents_used', 0)}")
                        print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
                        
                        print(f"\n Response:", flush=True)
                        print("-" * 50, flush=True)
                        print(result['answer'], flush=True)
                        print("-" * 50, flush=True)
                        
                        if result.get('reasoning_chain'):
                            print(f"\n Reasoning Chain:")
                            for i, step in enumerate(result['reasoning_chain'], 1):
                                print(f"   {i}. {step}")
                        
                        # Add to conversation history
                        conversation_history.append({
                            'query': query,
                            'response': result['answer']
                        })
                    else:
                        result = pipeline.query(
                            user_query=query,
                            conversation_history=conversation_history
                        )
                        
                        print(f"\n Response:", flush=True)
                        print("-" * 50, flush=True)
                        print(result['response'], flush=True)
                        print("-" * 50, flush=True)
                        
                        # Add to conversation history
                        conversation_history.append({
                            'query': query,
                            'response': result['response']
                        })
                    
                    # Keep only last 5 turns
                    if len(conversation_history) > 5:
                        conversation_history = conversation_history[-5:]
                        
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    print(f"\n Error: {str(e)}")
                    if "not aligned" in str(e):
                        print(" Tip: Try running with --reset_db to clear incompatible embeddings.")
                    
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                continue
    
    # Close connections before exiting
    pipeline.close_connections()


if __name__ == "__main__":
    main()
