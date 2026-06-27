class ContextBuilder:

    def build(
        self,
        query: str,
        documents: list[str],
    ) -> str:

        context = "\n\n".join(
            [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents)]
        )

        return f"""
                You are a helpful assistant.

                Answer ONLY using the provided context.

                If the answer cannot be found in the context,
                say that the information is not available.

                CONTEXT:

                {context}

                QUESTION:

                {query}

                ANSWER:
              """
