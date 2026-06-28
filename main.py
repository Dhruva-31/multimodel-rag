from pipelines.ingestion_pipeline import build_knowledge_base
from pipelines.query_pipeline import ask_question


def main():

    # =====================================
    # CHANGE THIS TO YOUR TEST FILE
    # =====================================

    file_path = "uploads/sample.pdf"
    # file_path = "uploads/sample.txt"
    # file_path = "uploads/sample.docx"

    print()
    print("=" * 60)
    print("BUILDING KNOWLEDGE BASE")
    print("=" * 60)

    bm25 = build_knowledge_base(file_path, "sample.pdf")

    print()
    print("Knowledge base ready.")

    while True:

        print()
        query = input("Query (or 'exit'): ")

        if query.lower() == "exit":
            break

        print()
        print("=" * 60)
        print("GENERATING RESPONSE")
        print("=" * 60)

        response = ask_question(
            query,
            bm25,
        )

        print()
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(response)
        print("=" * 60)


if __name__ == "__main__":
    main()
