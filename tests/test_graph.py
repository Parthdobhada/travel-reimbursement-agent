from dataclasses import dataclass
from typing import Dict, List

from agent.graph import TravelReimbursementAgent, build_graph
from agent.nodes import AgentNodes
from agent.state import create_initial_state


@dataclass
class FakeDocument:
    page_content: str
    metadata: Dict[str, str]


class FakePolicyRetriever:
    def __init__(self, context: str) -> None:
        self.context = context

    def execute(self, claim: Dict) -> Dict:
        return {
            "query": claim.get("description", ""),
            "policy_context": self.context,
            "retrieved_documents": [
                FakeDocument(
                    page_content=self.context,
                    metadata={
                        "section": "Meal Reimbursement Policy",
                        "policy_id": "POL-MEAL-001",
                    },
                )
            ],
            "retrieval_status": "success",
        }


def build_test_agent(
    policy_context: str = "POL-MEAL-001 Dinner limit is $75.",
    history: List[Dict] | None = None,
) -> TravelReimbursementAgent:
    nodes = AgentNodes(
        policy_retriever=FakePolicyRetriever(policy_context),
        llm=None,
        historical_claims=history or [],
    )
    return TravelReimbursementAgent(nodes=nodes)


def test_graph_compiles():
    nodes = AgentNodes(
        policy_retriever=FakePolicyRetriever("POL-MEAL-001 Limit is $75."),
        llm=None,
    )

    graph = build_graph(nodes)

    assert graph is not None


def test_graph_returns_partial_approval_when_limit_exceeded():
    agent = build_test_agent()
    result = agent.process_claim(
        {
            "claim_id": "CLM-PARTIAL",
            "employee_id": "EMP001",
            "expense_type": "meal",
            "amount": 95,
            "currency": "USD",
            "expense_date": "2026-06-20",
            "description": "Dinner during client meeting in London.",
            "receipt_uploaded": True,
        }
    )

    assert result["decision"] == "Partial Approval"
    assert result["approved_amount"] == 75
    assert result["rejected_amount"] == 20
    assert "POL-MEAL-001" in result["policy_references"]


def test_graph_routes_missing_receipt_to_manual_review():
    agent = build_test_agent()
    result = agent.process_claim(
        {
            "claim_id": "CLM-REVIEW",
            "employee_id": "EMP001",
            "expense_type": "meal",
            "amount": 45,
            "currency": "USD",
            "expense_date": "2026-06-20",
            "description": "Lunch during client meeting in London.",
            "receipt_uploaded": False,
        }
    )

    assert result["decision"] == "Manual Review"
    assert result["reviewer_required"] is True
    assert "Itemized receipt" in result["missing_documents"]
    assert "RC001" in result["reason_codes"]


def test_graph_state_contains_final_output():
    nodes = AgentNodes(
        policy_retriever=FakePolicyRetriever("POL-MEAL-001 Limit is $75."),
        llm=None,
    )
    graph = build_graph(nodes)

    final_state = graph.invoke(
        create_initial_state(
            {
                "claim_id": "CLM-STATE",
                "employee_id": "EMP001",
                "expense_type": "meal",
                "amount": 50,
                "currency": "USD",
                "expense_date": "2026-06-20",
                "description": "Lunch during client meeting in London.",
                "receipt_uploaded": True,
            }
        )
    )

    assert final_state["final_output"]["claim_id"] == "CLM-STATE"
    assert final_state["final_output"]["decision"] == "Approve"
    assert final_state["final_output"]["audit_trail"]["policy_retrieved"] is True
