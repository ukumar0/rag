class ContextDistiller:
    def __init__(self, llm, max_length=4000):
        self.llm = llm
        self.max_length = max_length

    def distill_context(self, documents, query):
        if self._calculate_length(documents) <= self.max_length:
            return documents

        summaries = []
        for doc in documents:
            summary = self.llm.generate(
                f"Summarize this document in relation to '{query}':\n\n{doc}"
            )
            summaries.append(summary)

        return summaries if self._calculate_length(summaries) <= self.max_length else summaries[:3]

    def _calculate_length(self, docs):
        return sum(len(doc) for doc in docs)
