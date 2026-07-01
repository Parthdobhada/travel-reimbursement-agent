from tools.policy_retriever import PolicyRetrieverTool

tool = PolicyRetrieverTool()

sample_claim = {

    "expense_type": "Hotel",

    "description":
        "Stayed at Marriott Hotel in Bangalore for two nights during client visit."

}

result = tool.execute(sample_claim)

print("=" * 60)

print(result["query"])

print("=" * 60)

print(result["policy_context"][:1500])