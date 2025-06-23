from typing import List, Dict, Any
import numpy as np

class MultiStepReasoner:
    def __init__(self, llm, retriever, db_manager=None, dense_retriever=None):
        self.llm = llm
        self.retriever = retriever
        self.db_manager = db_manager
        self.dense_retriever = dense_retriever

    def reason_step_by_step(self, query: str) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        
        # Decompose query into sub-questions
        logger.info(f"\n{'='*80}")
        logger.info(f" STARTING MULTI-STEP REASONING FOR: {query}")
        logger.info(f"{'='*80}")
        
        sub_questions = self._decompose_query(query)
        
        logger.info(f"\nGENERATED SUB-QUESTIONS:")
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info(f"   {i}. {sub_q}")
        
        chain = []
        context = ""
        intermediate_answers = {}

        # Answer each sub-question
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f" PROCESSING SUB-QUESTION {i}: {sub_q}")
            logger.info(f"{'─'*60}")
            
            # Prepare enhanced sub-query with context
            enhanced_sub_query = sub_q + " " + context
            logger.info(f" ENHANCED SUB-QUERY: {enhanced_sub_query[:200]}...")
            
            # Validate enhanced sub-query before using it
            enhanced_sub_query = enhanced_sub_query.strip()
            if not enhanced_sub_query:
                enhanced_sub_query = sub_q.strip()
                if not enhanced_sub_query:
                    enhanced_sub_query = query.strip()  # fallback to original query
                    if not enhanced_sub_query:
                        logger.warning(f"All queries are empty, skipping sub-question {i}")
                        continue
            
            # Get relevant documents for sub-question
            if self.db_manager:
                # Use database retrieval
                logger.info(" RETRIEVING FROM DATABASE (SQLite)")
                query_embedding = self.dense_retriever._get_embeddings([enhanced_sub_query])[0]
                docs = self.db_manager.search_similar_chunks(query_embedding, top_k=3)
                logger.info(f" RETRIEVED {len(docs)} DOCUMENTS FROM DATABASE")
            else:
                # Use in-memory retrieval
                logger.info(" RETRIEVING FROM HYBRID RETRIEVER")
                docs = self.retriever.retrieve(enhanced_sub_query)
                logger.info(f"RETRIEVED {len(docs)} DOCUMENTS FROM HYBRID RETRIEVER")
            
            # Log retrieved documents
            logger.info(f"\n RETRIEVED DOCUMENTS FOR SUB-QUESTION {i}:")
            for j, doc in enumerate(docs, 1):
                doc_preview = doc['content'][:150].replace('\n', ' ')
                logger.info(f"   Doc {j}: {doc_preview}...")
                if 'metadata' in doc and 'filename' in doc['metadata']:
                    logger.info(f"   Source: {doc['metadata']['filename']}")
            
            doc_context = "\n".join(doc['content'] for doc in docs)

            # Generate answer for sub-question
            step_prompt = f"""Sub-question {i}: {sub_q}

Available Context:
{doc_context}

Previous Knowledge:
{context}

Instructions:
1. Analyze the context and previous knowledge
2. Answer the sub-question
3. Explain your reasoning
4. Note any uncertainties

Format your response as:
Answer: [concise answer]
Reasoning: [explanation]
Confidence: [HIGH/MEDIUM/LOW]"""
            
            logger.info(f"\n SENDING PROMPT TO LLM:")
            logger.info(f"{'─'*40}")
            logger.info(step_prompt[:500] + "..." if len(step_prompt) > 500 else step_prompt)
            logger.info(f"{'─'*40}")
            
            step_response = self.llm.generate(step_prompt)
            
            logger.info(f"\n LLM RESPONSE FOR SUB-QUESTION {i}:")
            logger.info(f"{'─'*40}")
            logger.info(step_response)
            logger.info(f"{'─'*40}")
            
            chain.append(step_response)
            intermediate_answers[sub_q] = step_response
            context += f"\nStep {i}: {step_response}"

        # Synthesize final answer
        final_answer = self.llm.generate(
            f"""Original Question: {query}

Sub-questions and Answers:
{self._format_intermediate_answers(intermediate_answers)}

Task:
1. Synthesize a comprehensive answer to the original question
2. Ensure all aspects are addressed
3. Maintain logical flow
4. Acknowledge any uncertainties

Format:
1. Start with a clear, direct answer
2. Follow with supporting details
3. End with any caveats or limitations"""
        )

        return {
            "answer": final_answer,
            "reasoning_chain": chain
        }

    def _decompose_query(self, query: str) -> List[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate input query
        if not query or not query.strip():
            logger.warning("Empty query provided to decompose_query")
            return ["What information are you looking for?"]  # fallback question
        
        query = query.strip()
        
        decomp_prompt = f"""Break down this query into logical sub-questions:

Query: {query}

Requirements:
1. Each sub-question should be self-contained
2. Questions should build upon each other
3. Cover all aspects of the original query
4. Avoid redundancy
5. Maximum 3-4 sub-questions

Format:
- Sub-question 1: [question]
- Sub-question 2: [question]
..."""

        logger.info(f"\n SENDING DECOMPOSITION PROMPT TO OPENAI:")
        logger.info(f"{'─'*50}")
        logger.info(decomp_prompt)
        logger.info(f"{'─'*50}")

        response = self.llm.generate(decomp_prompt)
        
        logger.info(f"\n OPENAI DECOMPOSITION RESPONSE:")
        logger.info(f"{'─'*50}")
        logger.info(response)
        logger.info(f"{'─'*50}")
        
        # Extract sub-questions (basic parsing for PoC)
        sub_questions = []
        for line in response.split('\n'):
            if line.strip().startswith('- Sub-question'):
                question = line.split(':', 1)[1].strip()
                if question:  # Only add non-empty questions
                    sub_questions.append(question)
        
        # If parsing fails or no sub-questions found, return original query
        result = sub_questions if sub_questions else [query]
        
        # Filter out any empty questions
        result = [q.strip() for q in result if q.strip()]
        
        # Ensure we have at least one valid question
        if not result:
            result = [query]
        
        return result

    def _format_intermediate_answers(self, answers: Dict[str, str]) -> str:
        formatted = []
        for q, a in answers.items():
            formatted.append(f"Q: {q}\nA: {a}\n")
        return "\n".join(formatted)
