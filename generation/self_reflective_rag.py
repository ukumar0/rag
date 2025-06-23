class SelfReflectiveRAG:
    def __init__(self, llm, retriever, reflection_template):
        self.llm = llm
        self.retriever = retriever
        self.template = reflection_template
        self.max_context_length = 2000

    def should_retrieve(self, query, context=""):
        prompt = f"""Analyze if more information is needed to answer this query:

Query: '{query}'

Current Context:
{context[:500]}... [truncated]

Consider:
1. Is the current context sufficient?
2. Are there gaps in knowledge?
3. Would additional documents help?
4. Is the information up-to-date?

Return ONLY 'RETRIEVE' or 'SKIP' based on your analysis."""

        decision = self.llm.generate(prompt)
        return "RETRIEVE" in decision.upper()

    def generate_with_reflection(self, query, max_iterations=3):
        context = ""
        iterations = []
        
        for i in range(max_iterations):
            # Check if we need more information
            if self.should_retrieve(query, context):
                docs = self.retriever.retrieve(query)
                new_context = self._format_docs(docs)
                context = self._update_context(context, new_context)

            # Generate response
            response = self.llm.generate(
                f"""Query: {query}
                
Current Context:
{context}

Previous Attempts:
{self._format_iterations(iterations)}

Generate a comprehensive answer that addresses all aspects of the query.
If uncertain about any part, explicitly state it."""
            )
            
            # Evaluate response
            evaluation = self._evaluate_response(response, query, context)
            iterations.append({
                "response": response,
                "evaluation": evaluation
            })
            
            # Check if response is adequate
            if evaluation["is_adequate"]:
                break
                
            # Add reflection for next iteration
            context += f"\nReflection: {evaluation['reflection']}\n"
        
        return self._select_best_response(iterations)

    def _format_docs(self, docs):
        formatted = "\n".join([f"Document: {d}" for d in docs])
        return formatted[:self.max_context_length]

    def _update_context(self, old_context, new_context):
        combined = f"{old_context}\n{new_context}"
        if len(combined) > self.max_context_length:
            # Keep the most recent context if we exceed the limit
            return combined[-self.max_context_length:]
        return combined

    def _format_iterations(self, iterations):
        if not iterations:
            return "None"
        return "\n".join([f"Attempt {i+1}: {it['response'][:200]}..." 
                         for i, it in enumerate(iterations)])

    def _evaluate_response(self, response, query, context):
        prompt = f"""Evaluate this response:

Query: {query}

Response: {response}

Evaluate based on:
1. Completeness (addresses all aspects)
2. Accuracy (supported by context)
3. Clarity (well-structured and clear)
4. Relevance (focused on query)

Return a JSON-like structure:
{{
    "is_adequate": true/false,
    "scores": {{
        "completeness": 0-10,
        "accuracy": 0-10,
        "clarity": 0-10,
        "relevance": 0-10
    }},
    "reflection": "What could be improved...",
    "missing_aspects": ["aspect1", "aspect2"]
}}"""

        eval_result = self.llm.generate(prompt)
        try:
            is_adequate = "true" in eval_result.lower()
            return {
                "is_adequate": is_adequate,
                "reflection": eval_result,
                "response": response
            }
        except:
            return {
                "is_adequate": False,
                "reflection": "Failed to evaluate response properly",
                "response": response
            }

    def _select_best_response(self, iterations):
        if not iterations:
            return "No response generated"
            
        for it in reversed(iterations):
            if it["evaluation"]["is_adequate"]:
                return it["response"]
        return iterations[-1]["response"]




reflection_template = """You are an expert assistant.

Query: "{query}"
Context: "{context}"

Should you retrieve external documents to answer this question?
Respond with RETRIEVE or SKIP."""
