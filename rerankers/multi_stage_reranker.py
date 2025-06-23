from openai import OpenAI
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from config import OPENAI_API_KEY

class BaseReranker:
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

class RelevanceReranker(BaseReranker):
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = "text-embedding-3-small"
        
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
        
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        # DEBUG: Show what documents we received
        print(f"DEBUG RERANKER: Received {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):  # Show first 3
            content = doc.get('content', '')
            print(f"DEBUG RERANKER: Doc {i}: content_len={len(content)}, preview='{content[:100]}'")
            
        # Filter out documents with empty content BEFORE extracting texts
        valid_docs = [doc for doc in docs if doc.get('content', '').strip()]
        if not valid_docs:
            print("DEBUG RERANKER: No valid documents after filtering empty content")
            return []
            
        print(f"DEBUG RERANKER: {len(valid_docs)} valid documents after filtering")
        
        # Extract text content from valid documents only
        texts = [doc['content'] for doc in valid_docs]
            
        # Compute embeddings
        query_embedding = self._get_embeddings([query])[0]
        doc_embeddings = self._get_embeddings(texts)
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        return [valid_docs[i] for i in ranked_indices[:top_k]]

class DiversityReranker(BaseReranker):
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = "text-embedding-3-small"
        
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
        
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        # DEBUG: Show what documents we received
        print(f"DEBUG DIVERSITY RERANKER: Received {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):  # Show first 3
            content = doc.get('content', '')
            print(f"DEBUG DIVERSITY RERANKER: Doc {i}: content_len={len(content)}, preview='{content[:100]}'")
            
        # Filter out documents with empty content BEFORE extracting texts
        valid_docs = [doc for doc in docs if doc.get('content', '').strip()]
        if not valid_docs:
            print("DEBUG DIVERSITY RERANKER: No valid documents after filtering empty content")
            return []
            
        print(f"DEBUG DIVERSITY RERANKER: {len(valid_docs)} valid documents after filtering")
        
        # Extract text content from valid documents only
        texts = [doc['content'] for doc in valid_docs]
            
        # Compute embeddings
        embeddings = self._get_embeddings(texts)
        selected_indices = []
        
        # Select first document (highest relevance from previous stage)
        selected_indices.append(0)
        
        # Iteratively select diverse documents
        while len(selected_indices) < min(top_k, len(valid_docs)):
            max_min_sim = -1
            next_idx = -1
            
            # For each candidate document
            for i in range(len(valid_docs)):
                if i in selected_indices:
                    continue
                    
                # Compute minimum similarity to already selected docs
                min_sim = float('inf')
                for j in selected_indices:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    min_sim = min(min_sim, sim)
                
                # Select the document with highest minimum similarity
                if min_sim > max_min_sim:
                    max_min_sim = min_sim
                    next_idx = i
            
            if next_idx != -1:
                selected_indices.append(next_idx)
            else:
                break
        
        return [valid_docs[i] for i in selected_indices]

class FreshnessReranker(BaseReranker):
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        doc_dates: List[Tuple[Dict[str, Any], float]] = []
        now = datetime.now().timestamp()
        
        for doc in docs:
            # Check if document has actual timestamp in metadata
            if 'created_at' in doc.get('metadata', {}):
                try:
                    doc_timestamp = datetime.fromisoformat(doc['metadata']['created_at']).timestamp()
                except:
                    # If parsing fails, simulate age
                    age = np.random.randint(0, 90 * 24 * 3600)
                    doc_timestamp = now - age
            else:
                # Simulate document age (0 to 90 days old)
                age = np.random.randint(0, 90 * 24 * 3600)
                doc_timestamp = now - age
            
            doc_dates.append((doc, doc_timestamp))
            
        # Sort by timestamp (newest first)
        sorted_docs = sorted(doc_dates, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]

class AuthorityReranker(BaseReranker):
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        scored_docs: List[Tuple[Dict[str, Any], float]] = []
        authority_keywords = ['research', 'study', 'analysis', 'evidence', 'expert', 'official', 
                             'documentation', 'guide', 'manual', 'specification']
        
        for doc in docs:
            text = doc['content']
            # Score based on length (longer docs might be more authoritative)
            length_score = min(len(text.split()) / 100, 1.0)
            
            # Score based on presence of authority keywords
            keyword_score = sum(1 for keyword in authority_keywords if keyword.lower() in text.lower()) / len(authority_keywords)
            
            # Score based on source type if available in metadata
            source_score = 0.0
            if 'source' in doc.get('metadata', {}):
                source = doc['metadata']['source'].lower()
                if 'official' in source or 'documentation' in source:
                    source_score = 0.3
                elif 'manual' in source or 'guide' in source:
                    source_score = 0.2
            
            # Combined score
            score = 0.5 * length_score + 0.3 * keyword_score + 0.2 * source_score
            scored_docs.append((doc, score))
        
        # Sort by score
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]

class MultiStageReranker:
    def __init__(self):
        self.stages = [
            ("relevance", RelevanceReranker(), 0.4),
            ("diversity", DiversityReranker(), 0.3),
            ("freshness", FreshnessReranker(), 0.2),
            ("authority", AuthorityReranker(), 0.1)
        ]

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        current_docs = documents
        stage_scores = {}
        
        # Initialize scores using document content hash as key
        for doc in documents:
            content_hash = hash(doc['content'])
            stage_scores[content_hash] = 0.0
        
        # Apply each reranking stage
        for stage_name, reranker, weight in self.stages:
            # Get rankings from current stage
            stage_results = reranker.rerank(query, current_docs, len(current_docs))
            
            # Update scores based on rank
            for rank, doc in enumerate(stage_results):
                content_hash = hash(doc['content'])
                rank_score = 1.0 - (rank / len(stage_results))
                stage_scores[content_hash] += weight * rank_score
        
        # Create mapping from content hash to document
        doc_map = {hash(doc['content']): doc for doc in documents}
        
        # Sort by final scores
        final_ranking = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content_hash] for content_hash, _ in final_ranking[:top_k]]
