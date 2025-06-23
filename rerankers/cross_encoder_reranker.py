from openai import OpenAI
from typing import List, Dict, Any
from config import OPENAI_API_KEY
import re

class CrossEncoderReranker:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def _score_relevance(self, query: str, document_content: str) -> float:
        """Score relevance using OpenAI LLM"""
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.

Query: {query}

Document: {document_content}

Consider:
- Direct relevance to the query
- Quality of information
- Completeness of answer

Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract numeric score using regex
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            else:
                return 0.5  # Default score if parsing fails
                
        except Exception as e:
            print(f"Error scoring relevance: {e}")
            return 0.5  # Default score on error

    def _batch_score_relevance(self, query: str, documents: List[str]) -> List[float]:
        """Score multiple documents in batches for efficiency"""
        scores = []
        
        # For smaller batches, we can use a single prompt
        if len(documents) <= 5:
            doc_list = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
            prompt = f"""Rate the relevance of each document to the query on a scale of 0.0 to 1.0.

Query: {query}

{doc_list}

Respond with only the scores separated by commas (e.g., 0.85, 0.92, 0.73, 0.45, 0.91)"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50
                )
                
                score_text = response.choices[0].message.content.strip()
                # Extract all numeric scores
                score_matches = re.findall(r'(\d+\.?\d*)', score_text)
                
                if len(score_matches) == len(documents):
                    scores = [min(max(float(s), 0.0), 1.0) for s in score_matches]
                else:
                    # Fallback to individual scoring
                    scores = [self._score_relevance(query, doc) for doc in documents]
                    
            except Exception as e:
                print(f"Error in batch scoring: {e}")
                scores = [self._score_relevance(query, doc) for doc in documents]
        else:
            # For larger batches, score individually
            scores = [self._score_relevance(query, doc) for doc in documents]
        
        return scores

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using OpenAI-based relevance scoring.
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'content' and 'metadata' keys
            top_k: Number of documents to return
            
        Returns:
            List of reranked document dictionaries
        """
        if not documents:
            return []
            
        # Extract document contents
        doc_contents = [doc['content'] for doc in documents]
        
        # Get relevance scores
        scores = self._batch_score_relevance(query, doc_contents)
        
        # Create scored documents
        scored_docs = [
            {"doc": doc, "score": score} 
            for doc, score in zip(documents, scores)
        ]
        
        # Sort by score (descending)
        sorted_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        
        # Return top k documents
        return [item["doc"] for item in sorted_docs[:top_k]]
