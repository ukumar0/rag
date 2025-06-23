class QueryTransformer:
    def __init__(self, llm):
        self.llm = llm
        self.strategies = {
            "decomposition": self._decompose_query,
            "expansion": self._expand_query,
            "clarification": self._clarify_query,
            "step_back": self._step_back_query
        }

    def transform_query(self, query, strategy="auto"):
        # Validate input query
        if not query or not query.strip():
            return "What information are you looking for?"  # fallback query
        
        query = query.strip()
        
        if strategy == "auto":
            strategy = self._select_best_strategy(query)
        
        transformed = self.strategies[strategy](query)
        
        # Ensure the transformed query is not empty
        if not transformed or not transformed.strip():
            return query  # fallback to original query if transformation fails
        
        return transformed.strip()

    def _select_best_strategy(self, query):
        prompt = f"""Analyze this query and select the best strategy:
Query: '{query}'

Available strategies:
1. Decomposition - for complex, multi-part queries needing breakdown
2. Expansion - for short, ambiguous queries needing context
3. Clarification - for vague queries needing specification
4. Step Back - for overly specific queries needing broader context

Return ONLY ONE of: decomposition, expansion, clarification, step_back"""
        
        strategy = self.llm.generate(prompt).strip().lower()
        return strategy if strategy in self.strategies else "expansion"

    def _decompose_query(self, query):
        prompt = f"""Break down this query into simpler sub-questions:
Query: '{query}'

Return 2-3 sub-questions that together help answer the main query.
Format: number each sub-question on a new line"""
        
        decomposed = self.llm.generate(prompt)
        return decomposed

    def _expand_query(self, query):
        prompt = f"""Expand this query with relevant context and keywords:
Query: '{query}'

Consider:
1. Domain-specific terminology
2. Related concepts
3. Potential ambiguities

Return an expanded version of the query."""
        
        return self.llm.generate(prompt)

    def _clarify_query(self, query):
        prompt = f"""Make this query more specific and clear:
Query: '{query}'

Identify and clarify:
1. Ambiguous terms
2. Missing context
3. Implicit assumptions

Return a clearer version of the query."""
        
        return self.llm.generate(prompt)

    def _step_back_query(self, query):
        prompt = f"""Take a broader perspective on this query:
Query: '{query}'

Consider:
1. Higher-level concepts
2. Underlying principles
3. General context

Return a broader version of the query."""
        
        return self.llm.generate(prompt)
