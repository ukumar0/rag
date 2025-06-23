from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import Document
import re
import spacy
from openai import OpenAI
import numpy as np
from config import OPENAI_API_KEY
from typing import List
from utils.html_cleaner import HTMLCleaner

class EnhancedChunker:
    def __init__(self, semantic_model="text-embedding-3-small"):
        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load OpenAI client for semantic boundary detection
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = semantic_model
        
        # Initialize HTML cleaner
        self.html_cleaner = HTMLCleaner()
        
        # Initialize chunking strategies
        self.chunking_strategies = {
            "sentence_window": SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            ),
            "hierarchical": HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            ),
            "semantic": SemanticSplitterNodeParser.from_defaults(
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
        }

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from OpenAI API"""
        if not texts:
            return []
        
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
            return []
        
        try:
            response = self.client.embeddings.create(
                    input=clean_texts,
                model=self.model_name
            )
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings
        except Exception as e:
            # Log the error with more context
            print(f"Error creating embeddings: {e}")
            print(f"Number of texts: {len(clean_texts)}")
            print(f"Sample text (first 200 chars): {clean_texts[0][:200] if clean_texts else 'None'}")
            raise

    def process_with_strategy(self, document_text, strategy, clean_html=True):
        """
        Process document text with the specified chunking strategy.
        
        Args:
            document_text: The text content to chunk
            strategy: The chunking strategy to use
            clean_html: Whether to clean HTML content before chunking
            
        Returns:
            List of text chunks
        """
        # Clean HTML content if requested and present
        if clean_html and self.html_cleaner.is_html_content(document_text):
            document_text = self.html_cleaner.clean_html(document_text)
        
        if strategy == "semantic":
            return self._semantic_chunking(document_text)
        elif strategy == "hybrid":
            return self._hybrid_chunking(document_text)
        elif strategy == "hierarchical":
            return self._hierarchical_chunking(document_text)
        elif strategy == "advanced_hierarchical":
            return self._advanced_hierarchical_chunking(document_text)
        else:
            document = Document(text=document_text)
            nodes = self.chunking_strategies[strategy].get_nodes_from_documents([document])
            return [node.text for node in nodes]

    def _hierarchical_chunking(self, text):
        """Standard hierarchical chunking using LlamaIndex"""
        document = Document(text=text)
        nodes = self.chunking_strategies["hierarchical"].get_nodes_from_documents([document])
        return [node.text for node in nodes]

    def _advanced_hierarchical_chunking(self, text):
        """
        Advanced hierarchical chunking that combines:
        1. Document-level structure detection
        2. Hierarchical size-based chunking
        3. Semantic refinement at each level
        """
        # Level 1: Document structure (sections, chapters, etc.)
        structural_chunks = self._split_on_structure(text)
        
        all_chunks = []
        
        for structural_chunk in structural_chunks:
            # Level 2: Apply hierarchical chunking to each structural section
            doc = Document(text=structural_chunk)
            hierarchical_nodes = self.chunking_strategies["hierarchical"].get_nodes_from_documents([doc])
            
            # Level 3: Apply semantic refinement to medium-sized chunks
            for node in hierarchical_nodes:
                chunk_text = node.text
                word_count = len(chunk_text.split())
                
                if word_count > 100:  # Apply semantic splitting to larger chunks
                    semantic_chunks = self._semantic_split(chunk_text)
                    all_chunks.extend(semantic_chunks)
                else:
                    all_chunks.append(chunk_text)
        
        return all_chunks

    def _semantic_chunking(self, text):
        # First pass: Split on clear structural boundaries
        initial_chunks = self._split_on_structure(text)
        
        # Second pass: Refine chunks based on semantic coherence
        refined_chunks = []
        for chunk in initial_chunks:
            if len(chunk.split()) > 100:  # Only refine larger chunks
                refined_chunks.extend(self._semantic_split(chunk))
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks

    def _split_on_structure(self, text):
        chunks = []
        # Split on structural markers (headers, lists, etc.)
        patterns = [
            r'\n\s*#{1,6}\s+',  # Markdown headers
            r'\n\s*\d+\.\s+',   # Numbered lists
            r'\n\s*[-*]\s+',    # Bullet points
            r'\n\s*\n',         # Double newlines
        ]
        
        current_chunk = []
        lines = text.split('\n')
        
        for line in lines:
            if any(re.match(pattern, '\n' + line) for pattern in patterns):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
            current_chunk.append(line)
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    def _semantic_split(self, text):
        if self.nlp is None:
            # Fallback to simple sentence splitting if spaCy is not available
            sentences = self._simple_sentence_split(text)
        else:
            # Use spaCy for initial sentence splitting
            doc = self.nlp(text)
            sentences = [str(sent) for sent in doc.sents]
        
        if len(sentences) <= 3:
            return [text]  # Too few sentences to split meaningfully
        
        # Convert sentences to embeddings for semantic analysis
        embeddings = self._get_embeddings(sentences)
        
        if len(embeddings) <= 1:
            return [text]
        
        # Calculate semantic similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Find natural break points (low similarity)
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim - std_sim
            
            break_points = [
                i for i, sim in enumerate(similarities)
                if sim < threshold
            ]
        else:
            break_points = []
        
        # Create chunks based on break points
        chunks = []
        start = 0
        for point in break_points:
            if point - start >= 2:  # Ensure minimum chunk size
                chunk = ' '.join(sentences[start:point + 1])
                chunks.append(chunk)
                start = point + 1
                
        # Add remaining text
        if start < len(sentences):
            chunks.append(' '.join(sentences[start:]))
            
        return chunks if chunks else [text]

    def _simple_sentence_split(self, text):
        """Simple sentence splitting fallback"""
        # Basic sentence splitting using periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _hybrid_chunking(self, text):
        # Enhanced hybrid approach that includes hierarchical chunking
        structural_chunks = self._split_on_structure(text)
        final_chunks = []
        
        for chunk in structural_chunks:
            word_count = len(chunk.split())
            
            if word_count > 500:
                # Use hierarchical chunking for very large chunks
                doc = Document(text=chunk)
                hierarchical_nodes = self.chunking_strategies["hierarchical"].get_nodes_from_documents([doc])
                
                # Further refine with semantic chunking if needed
                for node in hierarchical_nodes:
                    node_text = node.text
                    if len(node_text.split()) > 100:
                        final_chunks.extend(self._semantic_split(node_text))
                    else:
                        final_chunks.append(node_text)
                        
            elif word_count > 200:
                # Use semantic chunking for large structural chunks
                final_chunks.extend(self._semantic_split(chunk))
            else:
                # Use sentence window for smaller chunks
                try:
                    doc = Document(text=chunk)
                    window_chunks = self.chunking_strategies["sentence_window"].get_nodes_from_documents([doc])
                    final_chunks.extend([node.text for node in window_chunks])
                except:
                    # Fallback to the original chunk if processing fails
                    final_chunks.append(chunk)
                
        return final_chunks

    def get_available_strategies(self):
        """Return list of available chunking strategies"""
        return [
            "semantic",           # Custom semantic chunking
            "hierarchical",       # Standard LlamaIndex hierarchical
            "advanced_hierarchical", # Multi-level hierarchical + semantic
            "hybrid",            # Enhanced hybrid (structural + hierarchical + semantic)
            "sentence_window",   # Sentence window with overlap
        ]

    def is_html_content(self, text: str) -> bool:
        """
        Check if text contains HTML content.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains HTML, False otherwise
        """
        return self.html_cleaner.is_html_content(text)

    def clean_html_content(self, text: str) -> str:
        """
        Clean HTML content from text.
        
        Args:
            text: Text potentially containing HTML
            
        Returns:
            Cleaned plain text
        """
        return self.html_cleaner.clean_html(text)

    def get_html_cleaning_info(self) -> dict:
        """
        Get information about HTML cleaning capabilities.
        
        Returns:
            Dictionary with HTML cleaning information
        """
        return {
            'html_cleaning_available': True,
            'html_cleaner_class': self.html_cleaner.__class__.__name__,
            'supported_features': [
                'HTML tag removal',
                'HTML entity decoding',
                'Structure preservation',
                'Link extraction',
                'Image metadata extraction',
                'Fallback regex cleaning'
            ]
        }
