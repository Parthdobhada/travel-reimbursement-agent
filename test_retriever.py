from rag.retriever import PolicyRetriever


def test_policy_retriever_returns_context():
    retriever = PolicyRetriever()

    query = "What is the hotel reimbursement limit?"

    documents = retriever.retrieve(query)

    print("=" * 60)
    print("Retrieved Documents")
    print("=" * 60)

    for i, doc in enumerate(documents, start=1):

        print(f"\nResult {i}")

        print(doc.metadata)

        print("-" * 50)

        print(doc.page_content[:400])

    print("\n")

    print("=" * 60)
    print("LLM Context")
    print("=" * 60)

    context = retriever.format_context(documents)

    print(context)

    assert documents
    assert context
