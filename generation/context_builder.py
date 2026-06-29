class ContextBuilder:

    def build(
        self,
        query: str,
        documents: list[tuple[str, float, dict]],
        max_chars: int = 10000,
    ) -> str:

        context = ""
        current_chars = 0

        for text, _, metadata in documents:

            if metadata["type"] == "pdf":

                chunk = (
                    f"SOURCE: {metadata['filename']} (p.{metadata['page']})\n{text}\n\n"
                )

            elif metadata["type"] in ["txt", "docx"]:

                chunk = f"SOURCE: {metadata['filename']} (¶{metadata['paragraph']})\n{text}\n\n"

            else:

                chunk = f"SOURCE: {metadata['filename']}\n{text}\n\n"

            if current_chars + len(chunk) > max_chars:
                break

            context += chunk
            current_chars += len(chunk)

        return f"""You are a strict RAG assistant. Answer based ONLY on the context below. If the answer is absent, output EXACTLY: "The information is not available in the provided context."

# RULES & CITATIONS
- NEVER use outside knowledge or fabricate facts, citations, or page/paragraph numbers.
- Extract ALL relevant evidence for completeness.
- Synthesize findings naturally; avoid phrases like "The context states" or "The passage suggests".
- Cite inline immediately after a claim using the smallest supporting span. Consecutive claims from the same source may share a citation.
- Format: PDF: (file.pdf, p.4), TXT/DOCX: (file.txt, ¶12), IMAGE: (file.png). Do NOT put a sources section at the end.

# GUIDELINES
- Factual: Direct answer + supporting evidence.
- Analytical: Group ALL evidence logically and explain relationships.
- Extraction: Logically organize all matching examples without omitting relevant evidence.
- Comparison: Detail similarities and differences covering evidence from all sides.

# CONTEXT
{context}

# QUESTION
{query}

# ANSWER
"""
