from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RAGAgentState(dict):
    """Dictionary subclass to maintain agent state (query, retrieved docs, response)."""
    pass


class EnhancedRAGAgent:
    """
    Enhanced RAG Agent that orchestrates the entire RAG pipeline with intelligent
    technique selection and multi-step reasoning capabilities.
    """
    
    def __init__(self, pipeline):
        """
        Initialize the RAG Agent with a pipeline instance.
        
        Args:
            pipeline: AdvancedRAGPipeline instance containing all components
        """
        self.pipeline = pipeline
        
        # Extract components from pipeline
        self.llm = pipeline.llm
        self.query_transformer = pipeline.query_transformer
        self.hybrid_retriever = pipeline.hybrid_retriever
        self.reranker = pipeline.reranker
        self.cross_encoder_reranker = pipeline.cross_encoder_reranker
        self.context_distiller = pipeline.context_distiller
        self.multi_step_reasoner = pipeline.multi_step_reasoner
        self.self_reflective_rag = pipeline.self_reflective_rag
        
        # Initialize query router if not available
        if hasattr(pipeline, 'query_router'):
            self.query_router = pipeline.query_router
        else:
            self.query_router = None
            logger.warning("Query router not found in pipeline, using default routing")

    def _retrieve_with_strategy(self, query: str, strategy: str) -> List[Dict]:
        """
        Retrieve documents using the specified strategy with multi-vector capabilities.
        
        Args:
            query: The query to search for
            strategy: The retrieval strategy to use
            
        Returns:
            List of retrieved documents
        """
        try:
            # First, try to populate the multi-vector retriever with documents from Weaviate
            # if it hasn't been populated yet
            if self.hybrid_retriever.multi.documents == []:
                logger.info("Multi-vector retriever not populated, indexing documents from Weaviate...")
                if hasattr(self.pipeline, 'weaviate_manager'):
                    # Use a generic query to get all documents instead of empty string
                    weaviate_docs = self.pipeline.weaviate_manager.search_similar("document text content", top_k=1000, similarity_threshold=0.0)
                    if weaviate_docs:
                        # Convert Weaviate format to multi-vector format
                        formatted_docs = []
                        for doc in weaviate_docs:
                            formatted_docs.append({
                                'content': doc.get('content', ''),
                                'metadata': doc.get('metadata', {})
                            })
                        
                        # Index documents in all retrievers
                        logger.info(f"Indexing {len(formatted_docs)} documents in hybrid retriever...")
                        self.hybrid_retriever.index_documents(formatted_docs)
                        logger.info("Multi-vector indexing completed")
                    else:
                        logger.warning("No documents found in Weaviate for multi-vector indexing")
                        # Try alternative approach using database directly
                        try:
                            logger.info("Trying to get documents from database directly...")
                            if hasattr(self.pipeline, 'db_manager'):
                                db_chunks = self.pipeline.db_manager.get_all_chunks()
                                if db_chunks:
                                    formatted_docs = []
                                    for chunk in db_chunks[:100]:  # Limit to first 100 for performance
                                        formatted_docs.append({
                                            'content': chunk.get('content', ''),
                                            'metadata': {
                                                'doc_id': chunk.get('doc_id', ''),
                                                'chunk_index': chunk.get('chunk_index', 0),
                                                'filename': chunk.get('filename', '')
                                            }
                                        })
                                    
                                    logger.info(f"Indexing {len(formatted_docs)} documents from database in hybrid retriever...")
                                    self.hybrid_retriever.index_documents(formatted_docs)
                                    logger.info("Multi-vector indexing from database completed")
                                else:
                                    logger.warning("No documents found in database either")
                        except Exception as db_error:
                            logger.warning(f"Failed to get documents from database: {str(db_error)}")
            
            # Now use the hybrid retriever with different strategies
            if strategy == "hybrid_dense":
                # Emphasize dense retrieval for factual queries
                logger.debug("Using hybrid retrieval with dense emphasis")
                # Adjust weights to favor dense retrieval
                self.hybrid_retriever.update_weights({'dense': 0.7, 'sparse': 0.2, 'multi': 0.1})
                return self.hybrid_retriever.retrieve(query, top_k=10)
                
            elif strategy == "hybrid_reasoning":
                # Use balanced approach with reasoning filtering for analytical queries
                logger.debug("Using hybrid retrieval with reasoning approach")
                # Balanced weights for comprehensive retrieval
                self.hybrid_retriever.update_weights({'dense': 0.5, 'sparse': 0.3, 'multi': 0.2})
                initial_docs = self.hybrid_retriever.retrieve(query, top_k=15)
                # Then apply reasoning-based filtering/expansion
                return self._apply_reasoning_filter(query, initial_docs)
                
            elif strategy == "hybrid_multi":
                # Emphasize multi-vector retrieval for comparative queries
                logger.debug("Using hybrid retrieval with multi-vector emphasis")
                # Favor multi-vector for diverse perspectives
                self.hybrid_retriever.update_weights({'dense': 0.4, 'sparse': 0.2, 'multi': 0.4})
                return self.hybrid_retriever.retrieve(query, top_k=12)
                
            elif strategy == "hybrid_temporal":
                # Use sparse + multi-vector for temporal queries
                logger.debug("Using hybrid retrieval with temporal emphasis")
                # Favor sparse for keyword matching and multi-vector for context
                self.hybrid_retriever.update_weights({'dense': 0.3, 'sparse': 0.4, 'multi': 0.3})
                temporal_query = f"{query} timeline chronological order historical context"
                return self.hybrid_retriever.retrieve(temporal_query, top_k=10)
                
            else:
                # Default balanced hybrid retrieval
                logger.debug(f"Using default balanced hybrid retrieval for strategy: {strategy}")
                # Reset to default balanced weights
                self.hybrid_retriever.update_weights({'dense': 0.6, 'sparse': 0.3, 'multi': 0.1})
                return self.hybrid_retriever.retrieve(query, top_k=10)
                
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fallback to direct Weaviate retrieval
            try:
                logger.info("Falling back to direct Weaviate retrieval")
                if hasattr(self.pipeline, 'weaviate_manager'):
                    return self.pipeline.weaviate_manager.search_similar(query, top_k=5, similarity_threshold=0.6)
                else:
                    return []
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []

    def _apply_reasoning_filter(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Apply reasoning-based filtering for analytical queries."""
        try:
            # For analytical queries, prioritize documents with explanatory content
            analytical_keywords = ["because", "reason", "cause", "explanation", "mechanism", "process", "how", "why"]
            
            scored_docs = []
            for doc in docs:
                content = doc.get('content', '').lower()
                analytical_score = sum(1 for keyword in analytical_keywords if keyword in content)
                # Create a copy to avoid modifying the original
                doc_copy = doc.copy()
                doc_copy['analytical_score'] = analytical_score
                scored_docs.append(doc_copy)
            
            # Sort by analytical relevance, then by original score
            # Note: Weaviate scores are similarity scores (higher = better)
            scored_docs.sort(key=lambda x: (x.get('analytical_score', 0), x.get('score', 0)), reverse=True)
            return scored_docs[:10]
            
        except Exception as e:
            logger.warning(f"Reasoning filter failed: {str(e)}")
            return docs[:10]

    def _combine_and_deduplicate(self, doc_lists: List[List[Dict]]) -> List[Dict]:
        """Combine multiple document lists and remove duplicates."""
        try:
            seen_content = set()
            combined_docs = []
            
            for doc_list in doc_lists:
                for doc in doc_list:
                    content = doc.get('content', '')
                    # Use first 100 characters as a simple deduplication key
                    content_key = content[:100].strip()
                    
                    if content_key not in seen_content and content_key:
                        seen_content.add(content_key)
                        combined_docs.append(doc)
            
            # Sort by score and return top documents
            combined_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
            return combined_docs[:12]
            
        except Exception as e:
            logger.warning(f"Document combination failed: {str(e)}")
            # Return first list as fallback
            return doc_lists[0] if doc_lists else []

    def _enhanced_retrieve(self, state: RAGAgentState) -> RAGAgentState:
        """
        Enhanced retrieval with query transformation and multi-stage reranking.
        
        Args:
            state: Current agent state containing query
            
        Returns:
            Updated state with retrieved and reranked documents
        """
        try:
            query = state["query"]
            logger.info(f"Starting enhanced retrieval for query: {query[:100]}...")

            # Transform query for better retrieval
            if self.query_transformer:
                transformed_query = self.query_transformer.transform_query(query)
                state["transformed_query"] = transformed_query
                logger.debug(f"Query transformed: {transformed_query[:100]}...")
            else:
                transformed_query = query
                state["transformed_query"] = query

            # Route query if router is available
            retrieval_strategy = "hybrid_dense"  # Default strategy
            if self.query_router:
                try:
                    retrieval_strategy = self.query_router.route_query(transformed_query, state)
                    state["retrieval_strategy"] = retrieval_strategy
                    logger.info(f" Query routed to strategy: {retrieval_strategy}")
                except Exception as e:
                    logger.warning(f"Query routing failed, using default: {str(e)}")
                    retrieval_strategy = "hybrid_dense"

            # Retrieve documents based on strategy
            documents = self._retrieve_with_strategy(transformed_query, retrieval_strategy)
            state["retrieved_documents"] = documents
            logger.info(f"Retrieved {len(documents)} documents using {retrieval_strategy}")

            # Multi-stage reranking
            if self.reranker and documents:
                try:
                    reranked = self.reranker.rerank(transformed_query, documents)
                    logger.info(f"First-stage reranking completed: {len(reranked)} docs")
                except Exception as e:
                    logger.warning(f"Multi-stage reranking failed: {str(e)}")
                    reranked = documents
            else:
                reranked = documents

            # Cross-encoder reranking for final precision
            if self.cross_encoder_reranker and reranked:
                try:
                    cross_reranked = self.cross_encoder_reranker.rerank(transformed_query, reranked)
                    state["reranked_documents"] = cross_reranked
                    logger.info(f"Cross-encoder reranking completed: {len(cross_reranked)} docs")
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed: {str(e)}")
                    state["reranked_documents"] = reranked
            else:
                state["reranked_documents"] = reranked

            return state
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {str(e)}")
            # Fallback to basic hybrid retrieval
            try:
                logger.info("Falling back to basic hybrid retrieval")
                documents = self.hybrid_retriever.retrieve(state["query"], top_k=5)
                state["retrieved_documents"] = documents
                state["reranked_documents"] = documents
                state["transformed_query"] = state["query"]
                state["retrieval_strategy"] = "fallback_hybrid"
                return state
            except Exception as fallback_error:
                logger.error(f"Fallback hybrid retrieval also failed: {str(fallback_error)}")
                # Final fallback to direct Weaviate
                try:
                    logger.info("Final fallback to direct Weaviate retrieval")
                    if hasattr(self.pipeline, 'weaviate_manager'):
                        documents = self.pipeline.weaviate_manager.search_similar(state["query"], top_k=3, similarity_threshold=0.5)
                        state["retrieved_documents"] = documents
                        state["reranked_documents"] = documents
                        state["transformed_query"] = state["query"]
                        state["retrieval_strategy"] = "fallback_weaviate"
                        return state
                except Exception as final_error:
                    logger.error(f"All retrieval methods failed: {str(final_error)}")
                
                state["retrieved_documents"] = []
                state["reranked_documents"] = []
                state["transformed_query"] = state["query"]
                state["retrieval_strategy"] = "failed"
                return state

    def _enhanced_generate(self, state: RAGAgentState) -> RAGAgentState:
        """
        Enhanced generation with context distillation and multi-step reasoning.
        
        Args:
            state: Current agent state with retrieved documents
            
        Returns:
            Updated state with generated response and reasoning chain
        """
        try:
            query = state["query"]
            docs = state.get("reranked_documents", [])
            
            if not docs:
                logger.warning("No documents available for generation")
                state["response"] = "I don't have enough relevant information to answer your question."
                state["reasoning_chain"] = ["No relevant documents found"]
                return state

            logger.info(f"Starting enhanced generation with {len(docs)} documents")

            # Distill context if needed
            if self.context_distiller:
                try:
                    distilled_docs = self.context_distiller.distill_context(docs, query)
                    state["distilled_docs"] = distilled_docs
                    logger.info(f"Context distilled from {len(docs)} to {len(distilled_docs)} docs")
                    working_docs = distilled_docs
                except Exception as e:
                    logger.warning(f"Context distillation failed: {str(e)}")
                    working_docs = docs[:5]  # Use top 5 as fallback
                    state["distilled_docs"] = working_docs
            else:
                working_docs = docs[:5]  # Use top 5 documents
                state["distilled_docs"] = working_docs

            # Generate with multi-step reasoning
            if self.multi_step_reasoner:
                try:
                    # Create context from working documents
                    context = "\n\n".join([
                        f"Document {i+1}: {doc.get('content', '')[:1000]}..."
                        for i, doc in enumerate(working_docs)
                    ])
                    
                    # Use multi-step reasoning
                    result = self.multi_step_reasoner.reason_step_by_step(query)
                    
                    state["response"] = result.get("answer", "Unable to generate response")
                    state["reasoning_chain"] = result.get("reasoning_chain", [])
                    state["confidence"] = result.get("confidence", 0.0)
                    
                    logger.info("Multi-step reasoning generation completed")
                    
                except Exception as e:
                    logger.warning(f"Multi-step reasoning failed: {str(e)}")
                    # Fallback to basic generation
                    state = self._fallback_generate(state, working_docs)
            else:
                # Fallback to basic generation
                state = self._fallback_generate(state, working_docs)

            return state
            
        except Exception as e:
            logger.error(f"Error in enhanced generation: {str(e)}")
            state["response"] = "I encountered an error while generating the response. Please try again."
            state["reasoning_chain"] = [f"Generation error: {str(e)}"]
            return state

    def _fallback_generate(self, state: RAGAgentState, docs: List[Dict]) -> RAGAgentState:
        """
        Fallback generation method using basic LLM generation.
        
        Args:
            state: Current agent state
            docs: Documents to use for context
            
        Returns:
            Updated state with generated response
        """
        try:
            query = state["query"]
            context = "\n\n".join([
                f"Document {i+1}: {doc.get('content', '')[:800]}..."
                for i, doc in enumerate(docs[:3])
            ])
            
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

            response = self.llm.generate(prompt)
            state["response"] = response
            state["reasoning_chain"] = ["Used basic LLM generation with retrieved context"]
            
            logger.info("Fallback generation completed")
            return state
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {str(e)}")
            state["response"] = "I'm unable to generate a response at this time."
            state["reasoning_chain"] = [f"Fallback generation error: {str(e)}"]
            return state

    def process_query(self, query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a query using the Enhanced RAG Agent with intelligent orchestration.
        
        Args:
            query: User's query
            conversation_context: Optional conversation history
            
        Returns:
            Dict containing response and metadata about processing
        """
        try:
            logger.info(f"Processing query with Enhanced RAG Agent: {query[:100]}...")
            
            # Initialize agent state
            state = RAGAgentState()
            state["query"] = query
            state["conversation_context"] = conversation_context or []
            state["timestamp"] = str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
            
            # Enhanced retrieval phase
            state = self._enhanced_retrieve(state)
            
            # Enhanced generation phase  
            state = self._enhanced_generate(state)
            
            # Prepare response with metadata
            response = {
                "answer": state.get("response", "Unable to generate response"),
                "query": query,
                "transformed_query": state.get("transformed_query", query),
                "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
                "documents_retrieved": len(state.get("retrieved_documents", [])),
                "documents_reranked": len(state.get("reranked_documents", [])),
                "documents_used": len(state.get("distilled_docs", [])),
                "reasoning_chain": state.get("reasoning_chain", []),
                "confidence": state.get("confidence", 0.0),
                "processing_metadata": {
                    "agent_type": "EnhancedRAGAgent",
                    "components_used": [
                        "query_transformer" if self.query_transformer else None,
                        "query_router" if self.query_router else None,
                        "hybrid_retriever",
                        "multi_stage_reranker" if self.reranker else None,
                        "cross_encoder_reranker" if self.cross_encoder_reranker else None,
                        "context_distiller" if self.context_distiller else None,
                        "multi_step_reasoner" if self.multi_step_reasoner else None
                    ]
                }
            }
            
            # Clean up None values from components_used
            response["processing_metadata"]["components_used"] = [
                comp for comp in response["processing_metadata"]["components_used"] if comp
            ]
            
            logger.info("Query processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "answer": "I encountered an error while processing your query. Please try again.",
                "query": query,
                "error": str(e),
                "documents_retrieved": 0,
                "documents_reranked": 0,
                "documents_used": 0,
                "reasoning_chain": [f"Processing error: {str(e)}"],
                "confidence": 0.0
            }

    def run(self, query: str) -> RAGAgentState:
        """
        Legacy method for backward compatibility.
        
        Args:
            query: User's query
            
        Returns:
            RAGAgentState with processing results
        """
        logger.info("Using legacy run method, consider using process_query instead")
        
        result = self.process_query(query)
        
        # Convert to RAGAgentState for compatibility
        state = RAGAgentState()
        state["query"] = query
        state["response"] = result["answer"]
        state["reasoning_chain"] = result.get("reasoning_chain", [])
        
        return state
