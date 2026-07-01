"""
Prompt templates for Gemini reasoning in the LangGraph agent.

The prompts in this module are intentionally side-effect free. Graph nodes
will import these builders when they need Gemini to choose tools or produce
the final reimbursement decision.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from agent.state import AgentState

logger = logging.getLogger(__name__)


AVAILABLE_AGENT_TOOLS: Dict[str, str] = {
    "Receipt_Checker": (
        "Checks whether required receipt evidence is present and whether "
        "missing documents should trigger Manual Review."
    ),
    "Expense_Limit_Checker": (
        "Checks the claim amount against monetary limits found in retrieved "
        "policy context."
    ),
    "Approval_Checker": (
        "Checks whether manager, director, VP, or finance approval is needed "
        "for the claim."
    ),
    "Duplicate_Checker": (
        "Checks receipt ID, invoice number, vendor, date, amount, and currency "
        "against historical claims for duplicate submissions."
    ),
}

MANUAL_REVIEW_TRIGGERS = (
    "missing receipt",
    "missing approval that is explicitly required by policy",
    "no relevant policy context retrieved",
    "conflicting non-approve tool results that cannot be resolved by tool priority",
    "confidence below 70",
    "policy exception requested by the employee",
    "explicit manual review policy matched by the claim facts",
)

TOOL_SELECTION_SCHEMA: Dict[str, Any] = {
    "tools_to_call": ["Receipt_Checker", "Expense_Limit_Checker"],
    "manual_review_required": False,
    "reason": "Short reason for selected tools.",
}

APPENDIX_N_OUTPUT_SCHEMA: Dict[str, Any] = {
    "claim_id": "string",
    "claim_status": (
        "Pending | Processing | Approved | Partially Approved | Rejected | "
        "Manual Review | Completed"
    ),
    "decision": "Approve | Partial Approval | Reject | Manual Review",
    "confidence_score": 98,
    "retrieval_confidence": "97%",
    "approved_amount": 0.0,
    "rejected_amount": 0.0,
    "currency": "INR | USD",
    "policy_references": ["POL-HOTEL-001", "POL-MEAL-001"],
    "retrieved_policy_sections": [
        "Hotel Accommodation Policy",
        "Meal Reimbursement Policy",
    ],
    "rag_chunks_used": ["chunk_12", "chunk_18"],
    "reason_codes": ["RC001"],
    "missing_documents": ["Hotel Invoice"],
    "tools_used": [
        "Policy_Retriever",
        "Receipt_Checker",
        "Expense_Limit_Checker",
        "Approval_Checker",
        "Output_Validator",
    ],
    "processing_time_ms": 1250,
    "agent_version": "1.0.1-LangGraph",
    "reviewer_required": False,
    "explanation": (
        "Short explanation citing specific policy IDs and tool outputs."
    ),
    "audit_trail": {
        "policy_retrieved": True,
        "receipt_checked": True,
        "expense_limit_checked": True,
        "approval_checked": True,
        "validator_passed": True,
        "llm_reasoning_completed": True,
    },
}

SYSTEM_PROMPT = """
You are the reasoning engine inside an Enterprise AI Travel Reimbursement
Agent. You are not a chatbot.

Use only the submitted claim, validation result, retrieved company policy
context, and structured tool outputs. Do not rely on general knowledge when
policy evidence is available. If policy evidence is missing or ambiguous,
route the claim to Manual Review.
Do not choose Manual Review merely because the retrieved context mentions a
manual-review policy. The claim facts or tool outputs must match that policy.

Allowed decisions:
- Approve
- Partial Approval
- Reject
- Manual Review

Manual Review is required for:
{manual_review_triggers}

If receipt evidence is present, the claim is within the applicable limit, no
duplicate is detected, no approval is missing, and no policy exception is
requested, approve the claim.

Amount rules:
- Approve: approved_amount equals the claim amount; rejected_amount is 0.
- Partial Approval: approved_amount equals the eligible policy limit;
  rejected_amount equals the excess.
- Reject: approved_amount is 0; rejected_amount equals the full claim amount.
- Manual Review: approved_amount is 0; rejected_amount equals the full claim
  amount pending human review.
"""


def build_tool_selection_prompt() -> ChatPromptTemplate:
    """
    Build the prompt Gemini uses to select follow-up tools.

    Returns:
        ChatPromptTemplate that expects formatted claim, validation, policy,
        and tool metadata inputs.
    """

    logger.info("Building tool selection prompt.")

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """
Decide which tools the LangGraph workflow should call next.

Available tools:
{available_tools_json}

Claim:
{claim_json}

Validation result:
{validation_result_json}

Retrieved policy context:
{policy_context}

Return only valid JSON matching this schema:
{tool_selection_schema_json}
""",
            ),
        ]
    )


def build_decision_prompt() -> ChatPromptTemplate:
    """
    Build the prompt Gemini uses to generate the final decision JSON.

    Returns:
        ChatPromptTemplate that expects formatted claim, validation, policy,
        and tool result inputs.
    """

    logger.info("Building final decision prompt.")

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """
Produce the final reimbursement decision.

Decision rules:
- Retrieve and cite policy before deciding.
- Use tool outputs as structured evidence.
- Return Manual Review only for missing receipt, missing required approval,
  missing policy, conflicting non-approve evidence, policy exceptions,
  confidence below 70, or explicit manual-review policy matches.
- Do not return Manual Review for a normal compliant hotel, meal, or travel
  claim simply because the policy context contains generic escalation rules.
- Return Partial Approval only when policy supports approving a portion and
  rejecting the remainder.
- Return Approve when receipt evidence is present, amount is within limit, no
  duplicate exists, no required approval is missing, policy was retrieved, and
  no exception was requested.
- Include reason codes when available from policy or tool evidence.
- Keep the explanation short and cite specific policy IDs.

Claim:
{claim_json}

Validation result:
{validation_result_json}

Retrieved policy context:
{policy_context}

Tool results:
{tool_results_json}

Audit trail:
{audit_trail_json}

Return only valid JSON matching Appendix N:
{output_schema_json}
""",
            ),
        ]
    )


def build_tool_selection_inputs(state: AgentState) -> Dict[str, str]:
    """
    Convert AgentState into prompt inputs for tool selection.

    Args:
        state: Current LangGraph state.

    Returns:
        Stringified inputs accepted by build_tool_selection_prompt().
    """

    logger.info("Formatting tool selection prompt inputs.")

    return {
        "manual_review_triggers": format_json(MANUAL_REVIEW_TRIGGERS),
        "available_tools_json": format_json(AVAILABLE_AGENT_TOOLS),
        "claim_json": format_json(state.get("claim", {})),
        "validation_result_json": format_json(
            state.get("validation_result", {})
        ),
        "policy_context": state.get("policy_context", ""),
        "tool_selection_schema_json": format_json(TOOL_SELECTION_SCHEMA),
    }


def build_decision_inputs(state: AgentState) -> Dict[str, str]:
    """
    Convert AgentState into prompt inputs for final decision generation.

    Args:
        state: Current LangGraph state.

    Returns:
        Stringified inputs accepted by build_decision_prompt().
    """

    logger.info("Formatting final decision prompt inputs.")

    return {
        "manual_review_triggers": format_json(MANUAL_REVIEW_TRIGGERS),
        "claim_json": format_json(state.get("claim", {})),
        "validation_result_json": format_json(
            state.get("validation_result", {})
        ),
        "policy_context": state.get("policy_context", ""),
        "tool_results_json": format_json(state.get("tool_results", {})),
        "audit_trail_json": format_json(state.get("audit_trail", [])),
        "output_schema_json": format_json(APPENDIX_N_OUTPUT_SCHEMA),
    }


def format_json(value: Any) -> str:
    """
    Serialize prompt data with stable formatting.

    Args:
        value: JSON-serializable or string-convertible object.

    Returns:
        Pretty JSON string for prompt injection.
    """

    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        default=str,
    )
