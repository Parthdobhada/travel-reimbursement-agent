from tools.policy_retriever import PolicyRetrieverTool
from tools.expense_limit_checker import ExpenseLimitChecker

claim = {

    "expense_type": "Hotel",

    "description":
        "Stayed at Marriott Hotel in Bangalore.",

    "amount": 9000

}

retriever = PolicyRetrieverTool()

policy = retriever.execute(claim)

checker = ExpenseLimitChecker()

result = checker.execute(
    claim,
    policy["policy_context"]
)

print("=" * 60)

for k, v in result.items():

    print(f"{k}: {v}")