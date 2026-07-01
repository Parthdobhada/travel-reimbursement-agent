"""
LangGraph state definitions for the travel reimbursement agent.

This module defines the shared state object passed between graph nodes.
It intentionally contains no tool execution, retrieval, or LLM logic.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, NotRequired, TypedDict

logger = logging.getLogger(__name__)


Decision = Literal[
    "Approve",
    "Partial Approval",
    "Reject",
    "Manual Review",
]

ClaimStatus = Literal[
    "Pending",
    "Processing",
    "Approved",
    "Partially Approved",
    "Rejected",
    "Manual Review",
    "Completed",
]


class AuditEvent(TypedDict, total=False):
    """Single audit event emitted during claim processing."""

    step: str
    status: str
    message: str
    timestamp: str
    details: NotRequired[Dict[str, Any]]


class ToolResult(TypedDict, total=False):
    """Structured result returned by a graph tool node."""

    tool_name: str
    status: str
    decision: NotRequired[Decision]
    approved_amount: NotRequired[float]
    rejected_amount: NotRequired[float]
    reason: NotRequired[str]
    audit: NotRequired[Dict[str, Any]]


class AgentState(TypedDict, total=False):
    """
    Shared LangGraph state for evaluating a reimbursement claim.

    List fields use LangGraph-compatible reducers so independent nodes can
    append evidence without overwriting earlier results.
    """

    claim: Dict[str, Any]
    claim_status: ClaimStatus
    validation_result: Dict[str, Any]
    policy_context: str
    retrieved_documents: List[Any]
    tool_results: Dict[str, ToolResult]
    tools_to_call: List[str]
    llm_tool_selection: Dict[str, Any]
    decision: Decision
    approved_amount: float
    rejected_amount: float
    currency: str
    confidence_score: float
    retrieval_confidence: float
    reason_codes: List[str]
    missing_documents: List[str]
    policy_references: List[str]
    retrieved_policy_sections: List[str]
    rag_chunks_used: List[str]
    tools_used: List[str]
    reviewer_required: bool
    explanation: str
    final_output: Dict[str, Any]
    processing_started_at: float
    audit_trail: List[AuditEvent]
    processing_time_ms: int
    agent_version: str


def create_initial_state(claim: Dict[str, Any]) -> AgentState:
    """
    Create the starting state for a reimbursement claim.

    Args:
        claim: Raw claim payload submitted through JSON, CSV, form, or API.

    Returns:
        AgentState with safe defaults for downstream graph nodes.
    """

    claim_id = claim.get("claim_id", "unknown")

    logger.info("Creating initial agent state for claim: %s", claim_id)

    return {
        "claim": dict(claim),
        "claim_status": "Pending",
        "validation_result": {},
        "policy_context": "",
        "retrieved_documents": [],
        "tool_results": {},
        "tools_to_call": [],
        "llm_tool_selection": {},
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "currency": str(claim.get("currency", "")),
        "confidence_score": 0.0,
        "retrieval_confidence": 0.0,
        "reason_codes": [],
        "missing_documents": [],
        "policy_references": [],
        "retrieved_policy_sections": [],
        "rag_chunks_used": [],
        "tools_used": [],
        "reviewer_required": False,
        "explanation": "",
        "final_output": {},
        "audit_trail": [
            build_audit_event(
                step="claim_intake",
                status="received",
                message="Claim received and initial state created.",
                details={"claim_id": claim_id},
            )
        ],
        "processing_time_ms": 0,
        "agent_version": "1.0.0-LangGraph",
    }


def build_audit_event(
    step: str,
    status: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> AuditEvent:
    """
    Build a timestamped audit event.

    Args:
        step: Workflow step or node name.
        status: Outcome status for the step.
        message: Human-readable audit message.
        details: Optional structured context.

    Returns:
        AuditEvent suitable for appending to AgentState.audit_trail.
    """

    event: AuditEvent = {
        "step": step,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if details:
        event["details"] = details

    return event


def add_audit_event(
    state: AgentState,
    step: str,
    status: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> AgentState:
    """
    Return a new state with an additional audit event.

    Args:
        state: Current agent state.
        step: Workflow step or node name.
        status: Outcome status for the step.
        message: Human-readable audit message.
        details: Optional structured context.

    Returns:
        Updated AgentState.
    """

    updated_state: AgentState = dict(state)
    audit_trail = list(state.get("audit_trail", []))
    audit_trail.append(
        build_audit_event(
            step=step,
            status=status,
            message=message,
            details=details,
        )
    )
    updated_state["audit_trail"] = audit_trail

    logger.info("Audit event added for step: %s", step)

    return updated_state


def add_tool_result(
    state: AgentState,
    tool_name: str,
    result: ToolResult,
) -> AgentState:
    """
    Return a new state with a tool result recorded.

    Args:
        state: Current agent state.
        tool_name: Stable tool name used by graph nodes.
        result: Structured tool result.

    Returns:
        Updated AgentState.
    """

    updated_state: AgentState = dict(state)
    tool_results = dict(state.get("tool_results", {}))
    tools_used = list(state.get("tools_used", []))

    tool_results[tool_name] = {
        "tool_name": tool_name,
        **result,
    }

    if tool_name not in tools_used:
        tools_used.append(tool_name)

    updated_state["tool_results"] = tool_results
    updated_state["tools_used"] = tools_used

    logger.info("Tool result recorded: %s", tool_name)

    return add_audit_event(
        state=updated_state,
        step=tool_name,
        status=str(result.get("status", "completed")),
        message=f"{tool_name} completed.",
        details={"result": result},
    )
