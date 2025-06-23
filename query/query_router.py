class QueryRouter:
    def __init__(self):
        self.routes = {
            "factual": self._route_factual,
            "analytical": self._route_analytical,
            "comparative": self._route_comparative,
            "temporal": self._route_temporal
        }

    def route_query(self, query, context=None):
        """
        Route query to appropriate retrieval strategy based on query type.
        
        Args:
            query: The user's query
            context: Optional context from agent state
            
        Returns:
            String indicating the retrieval strategy to use
        """
        query_type = self._classify_query(query)
        return self.routes[query_type](query, context)

    def _classify_query(self, query):
        """
        Classify query type using keyword analysis and patterns.
        
        Args:
            query: The user's query
            
        Returns:
            String indicating query type (factual, analytical, comparative, temporal)
        """
        query_lower = query.lower()
        
        # Enhanced classification with more keywords
        comparative_keywords = ["compare", "versus", "vs", "difference", "better", "worse", "contrast", "similar", "unlike"]
        analytical_keywords = ["why", "how", "explain", "analyze", "reason", "cause", "because", "mechanism", "process"]
        temporal_keywords = ["when", "date", "time", "year", "ago", "before", "after", "during", "timeline", "history"]
        
        if any(keyword in query_lower for keyword in comparative_keywords):
            return "comparative"
        elif any(keyword in query_lower for keyword in analytical_keywords):
            return "analytical"
        elif any(keyword in query_lower for keyword in temporal_keywords):
            return "temporal"
        else:
            return "factual"

    def _route_factual(self, query, context):
        """Route factual queries to hybrid retrieval with emphasis on dense search."""
        return "hybrid_dense"

    def _route_analytical(self, query, context):
        """Route analytical queries to multi-step reasoning approach."""
        return "hybrid_reasoning"

    def _route_comparative(self, query, context):
        """Route comparative queries to multi-vector retrieval for diverse perspectives."""
        return "hybrid_multi"

    def _route_temporal(self, query, context):
        """Route temporal queries to dense retrieval with temporal awareness."""
        return "hybrid_temporal"
