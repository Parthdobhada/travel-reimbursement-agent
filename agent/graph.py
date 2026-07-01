"""
LangGraph workflow for the travel reimbursement agent.
"""

import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState, create_initial_state

logger = logging.getLogger(__name__)


def build_graph(nodes: AgentNodes | None = None) -> Any:
    """
    Build and compile the LangGraph reimbursement workflow.

    Args:
        nodes: Optional node dependency container.

    Returns:
        Compiled LangGraph application.
    """

    agent_nodes = nodes or AgentNodes()
    graph = StateGraph(AgentState)

    graph.add_node("validate_claim", agent_nodes.validate_claim)
    graph.add_node("retrieve_policy", agent_nodes.retrieve_policy)
    graph.add_node("gemini_decision", agent_nodes.gemini_decision)
    graph.add_node("receipt_checker", agent_nodes.receipt_checker_node)
    graph.add_node(
        "expense_limit_checker",
        agent_nodes.expense_limit_checker_node,
    )
    graph.add_node("approval_checker", agent_nodes.approval_checker_node)
    graph.add_node("duplicate_checker", agent_nodes.duplicate_checker_node)
    graph.add_node(
        "confidence_calculator",
        agent_nodes.confidence_calculator_node,
    )
    graph.add_node("final_decision", agent_nodes.final_decision_node)
    graph.add_node("explanation", agent_nodes.explanation_node)
    graph.add_node("output", agent_nodes.output_node)

    graph.add_edge(START, "validate_claim")
    graph.add_edge("validate_claim", "retrieve_policy")
    graph.add_edge("retrieve_policy", "gemini_decision")

    graph.add_conditional_edges(
        "gemini_decision",
        agent_nodes.route_after_tool_selection,
        {
            "receipt_checker": "receipt_checker",
            "expense_limit_checker": "expense_limit_checker",
            "approval_checker": "approval_checker",
            "duplicate_checker": "duplicate_checker",
            "confidence_calculator": "confidence_calculator",
        },
    )
    graph.add_conditional_edges(
        "receipt_checker",
        agent_nodes.route_after_receipt_checker,
        {
            "expense_limit_checker": "expense_limit_checker",
            "approval_checker": "approval_checker",
            "duplicate_checker": "duplicate_checker",
            "confidence_calculator": "confidence_calculator",
        },
    )
    graph.add_conditional_edges(
        "expense_limit_checker",
        agent_nodes.route_after_expense_limit_checker,
        {
            "approval_checker": "approval_checker",
            "duplicate_checker": "duplicate_checker",
            "confidence_calculator": "confidence_calculator",
        },
    )
    graph.add_conditional_edges(
        "approval_checker",
        agent_nodes.route_after_approval_checker,
        {
            "duplicate_checker": "duplicate_checker",
            "confidence_calculator": "confidence_calculator",
        },
    )
    graph.add_conditional_edges(
        "duplicate_checker",
        agent_nodes.route_after_duplicate_checker,
        {"confidence_calculator": "confidence_calculator"},
    )
    graph.add_conditional_edges(
        "confidence_calculator",
        agent_nodes.route_after_confidence,
        {
            "final_decision": "final_decision",
            "explanation": "explanation",
        },
    )
    graph.add_edge("final_decision", "explanation")
    graph.add_edge("explanation", "output")
    graph.add_edge("output", END)

    logger.info("Travel reimbursement LangGraph workflow compiled.")

    return graph.compile()


class TravelReimbursementAgent:
    """High-level interface for processing reimbursement claims."""

    def __init__(
        self,
        nodes: AgentNodes | None = None,
        historical_claims: List[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            nodes: Optional dependency-injected node container.
            historical_claims: Optional duplicate-check history.
        """

        self.nodes = nodes or AgentNodes(historical_claims=historical_claims)
        self.graph = build_graph(self.nodes)

    def process_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single claim through the LangGraph workflow.

        Args:
            claim: Raw claim payload from JSON, CSV, form, or API.

        Returns:
            Final structured reimbursement decision JSON.
        """

        initial_state = create_initial_state(claim)
        final_state = self.graph.invoke(initial_state)

        return final_state.get("final_output", final_state)
