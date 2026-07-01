"""
LangGraph node implementations for the reimbursement agent.

Every business capability is exposed as a node-compatible method. External
dependencies are injected through AgentNodes so tests can replace Gemini or RAG
without changing graph structure.
"""

import logging
import time
from typing import Any, Dict, List, Protocol

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.prompts import (
    AVAILABLE_AGENT_TOOLS,
    build_decision_inputs,
    build_decision_prompt,
    build_tool_selection_inputs,
    build_tool_selection_prompt,
)
from agent.state import AgentState, add_audit_event, add_tool_result
from config.settings import settings
from schemas.output_schema import ReimbursementDecision
from tools.approval_checker import ApprovalChecker
from tools.confidence_calculator import ConfidenceCalculator
from tools.duplicate_checker import DuplicateChecker
from tools.expense_limit_checker import ExpenseLimitChecker
from tools.explanation_generator import ExplanationGenerator
from tools.policy_retriever import PolicyRetrieverTool
from tools.receipt_checker import ReceiptChecker
from tools.validator import ClaimValidator
from utils.helpers import (
    extract_json_object,
    extract_policy_references,
    percentage,
    unique_preserve_order,
)

logger = logging.getLogger(__name__)

TOOL_ORDER = [
    "Receipt_Checker",
    "Expense_Limit_Checker",
    "Approval_Checker",
    "Duplicate_Checker",
]

_DEFAULT_LLM = object()


class LLMClient(Protocol):
    """Protocol for LangChain-compatible chat model clients."""

    def invoke(self, input: Any) -> Any:
        """Invoke the model."""


class AgentNodes:
    """Container for node dependencies and LangGraph node methods."""

    def __init__(
        self,
        validator: ClaimValidator | None = None,
        policy_retriever: PolicyRetrieverTool | None = None,
        receipt_checker: ReceiptChecker | None = None,
        expense_limit_checker: ExpenseLimitChecker | None = None,
        approval_checker: ApprovalChecker | None = None,
        duplicate_checker: DuplicateChecker | None = None,
        confidence_calculator: ConfidenceCalculator | None = None,
        explanation_generator: ExplanationGenerator | None = None,
        llm: LLMClient | None | object = _DEFAULT_LLM,
        historical_claims: List[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize graph node dependencies.

        Args:
            validator: Claim validation tool.
            policy_retriever: RAG policy retriever tool.
            receipt_checker: Receipt completeness tool.
            expense_limit_checker: Policy limit checker tool.
            approval_checker: Approval requirement checker.
            duplicate_checker: Duplicate detection tool.
            confidence_calculator: Confidence scoring tool.
            explanation_generator: Human-readable explanation tool.
            llm: Optional Gemini-compatible LLM client.
            historical_claims: Optional duplicate-check history.
        """

        self.validator = validator or ClaimValidator()
        self.policy_retriever = policy_retriever or PolicyRetrieverTool()
        self.receipt_checker = receipt_checker or ReceiptChecker()
        self.expense_limit_checker = (
            expense_limit_checker or ExpenseLimitChecker()
        )
        self.approval_checker = approval_checker or ApprovalChecker()
        self.duplicate_checker = duplicate_checker or DuplicateChecker()
        self.confidence_calculator = (
            confidence_calculator or ConfidenceCalculator()
        )
        self.explanation_generator = (
            explanation_generator or ExplanationGenerator()
        )
        self.llm = self._build_default_llm() if llm is _DEFAULT_LLM else llm
        self.historical_claims = historical_claims or []

    def validate_claim(self, state: AgentState) -> AgentState:
        """Validate raw claim data before retrieval or reasoning."""

        started_at = state.get("processing_started_at", time.perf_counter())
        result = self.validator.validate(state.get("claim", {}))

        updated_state: AgentState = {
            **state,
            "processing_started_at": started_at,
            "claim_status": "Processing",
            "validation_result": result,
        }

        if not result.get("is_valid"):
            updated_state["decision"] = "Manual Review"
            updated_state["reviewer_required"] = True
            updated_state["reason_codes"] = unique_preserve_order(
                list(state.get("reason_codes", [])) + ["RC008"]
            )
            updated_state["missing_documents"] = unique_preserve_order(
                list(state.get("missing_documents", []))
                + result.get("errors", [])
            )

        return add_audit_event(
            updated_state,
            step="Validator",
            status=result.get("status", "completed"),
            message="Claim validation completed.",
            details={"errors": result.get("errors", [])},
        )

    def retrieve_policy(self, state: AgentState) -> AgentState:
        """Retrieve policy context through the existing RAG tool."""

        if not state.get("validation_result", {}).get("is_valid", False):
            return add_audit_event(
                state,
                step="Policy_Retriever",
                status="skipped",
                message="Policy retrieval skipped because validation failed.",
            )

        result = self.policy_retriever.execute(state.get("claim", {}))
        documents = result.get("retrieved_documents", [])
        policy_context = result.get("policy_context", "")
        policy_references = self._document_policy_references(documents)

        updated_state: AgentState = {
            **state,
            "policy_context": policy_context,
            "retrieved_documents": documents,
            "policy_references": unique_preserve_order(
                list(state.get("policy_references", [])) + policy_references
            ),
            "retrieved_policy_sections": unique_preserve_order(
                list(state.get("retrieved_policy_sections", []))
                + [
                    doc.metadata.get("section", "Unknown Section")
                    for doc in documents
                ]
            ),
            "rag_chunks_used": unique_preserve_order(
                list(state.get("rag_chunks_used", []))
                + [
                    str(
                        doc.metadata.get(
                            "chunk_id",
                            doc.metadata.get("policy_id", "unknown"),
                        )
                    )
                    for doc in documents
                ]
            ),
            "retrieval_confidence": 95.0 if policy_context else 0.0,
            "tools_used": unique_preserve_order(
                list(state.get("tools_used", [])) + ["Policy_Retriever"]
            ),
        }

        if not policy_context:
            updated_state["decision"] = "Manual Review"
            updated_state["reviewer_required"] = True

        return add_audit_event(
            updated_state,
            step="Policy_Retriever",
            status=result.get("retrieval_status", "completed"),
            message="Policy retrieval completed.",
            details={"documents_retrieved": len(documents)},
        )

    def gemini_decision(self, state: AgentState) -> AgentState:
        """
        Ask Gemini which tools should run next, with deterministic fallback.
        """

        if self._requires_manual_review_before_llm(state):
            return self._record_tool_selection(
                state,
                tools_to_call=["Receipt_Checker", "Approval_Checker"],
                selection={
                    "manual_review_required": True,
                    "reason": "Validation or policy retrieval requires review.",
                },
            )

        selection = self._invoke_tool_selection_llm(state)
        tools_to_call = selection.get("tools_to_call") or []

        if not tools_to_call:
            tools_to_call = self._fallback_tool_selection(state)
            selection["reason"] = selection.get(
                "reason",
                "Fallback rules selected tools.",
            )

        return self._record_tool_selection(
            state,
            tools_to_call=tools_to_call,
            selection=selection,
        )

    def receipt_checker_node(self, state: AgentState) -> AgentState:
        """Run the receipt checker tool."""

        result = self.receipt_checker.execute(state.get("claim", {}))
        updated_state = self._merge_tool_evidence(state, result)
        return add_tool_result(updated_state, "Receipt_Checker", result)

    def expense_limit_checker_node(self, state: AgentState) -> AgentState:
        """Run the expense limit checker tool."""

        result = self.expense_limit_checker.execute(
            state.get("claim", {}),
            state.get("policy_context", ""),
        )
        result["tool_name"] = "Expense_Limit_Checker"
        updated_state = self._merge_tool_evidence(state, result)
        return add_tool_result(updated_state, "Expense_Limit_Checker", result)

    def approval_checker_node(self, state: AgentState) -> AgentState:
        """Run the approval checker tool."""

        result = self.approval_checker.execute(state.get("claim", {}))
        updated_state = self._merge_tool_evidence(state, result)
        return add_tool_result(updated_state, "Approval_Checker", result)

    def duplicate_checker_node(self, state: AgentState) -> AgentState:
        """Run the duplicate checker tool."""

        result = self.duplicate_checker.execute(
            state.get("claim", {}),
            historical_claims=self.historical_claims,
        )
        updated_state = self._merge_tool_evidence(state, result)
        return add_tool_result(updated_state, "Duplicate_Checker", result)

    def confidence_calculator_node(self, state: AgentState) -> AgentState:
        """
        Run confidence scoring after selected tools complete.
        """

        result = self.confidence_calculator.execute(state)
        confidence = float(result.get("confidence_score", 0.0))

        updated_state: AgentState = {
            **state,
            "confidence_score": confidence,
        }

        if result.get("status") == "manual_review":
            updated_state["decision"] = "Manual Review"
            updated_state["reviewer_required"] = True

        return add_tool_result(
            updated_state,
            "Confidence_Calculator",
            result,
        )

    def final_decision_node(self, state: AgentState) -> AgentState:
        """
        Produce the final decision using Gemini with deterministic fallback.
        """

        decision_payload = self._invoke_final_decision_llm(state)

        if not decision_payload:
            decision_payload = self._fallback_final_decision(state)

        updated_state = self._apply_decision_payload(state, decision_payload)

        return add_audit_event(
            updated_state,
            step="Gemini_Final_Decision",
            status="completed",
            message="Final decision generated.",
            details={"decision": updated_state.get("decision")},
        )

    def explanation_node(self, state: AgentState) -> AgentState:
        """Generate a concise explanation from final graph evidence."""

        decision = self._business_decision_from_evidence(
            state,
            state.get("decision"),
        )
        approved_amount, rejected_amount = self._amounts_for_decision(
            state,
            decision,
        )
        normalized_state: AgentState = {
            **state,
            "decision": decision,
            "approved_amount": approved_amount,
            "rejected_amount": rejected_amount,
            "reviewer_required": decision == "Manual Review",
            "policy_references": self._policy_references_for_decision(
                state,
                decision,
            ),
        }

        result = self.explanation_generator.execute(normalized_state)
        updated_state: AgentState = {
            **normalized_state,
            "explanation": result.get("explanation", state.get("explanation", "")),
        }

        return add_tool_result(
            updated_state,
            "Explanation_Generator",
            result,
        )

    def output_node(self, state: AgentState) -> AgentState:
        """Validate and return Appendix-N-style structured JSON."""

        started_at = float(state.get("processing_started_at", time.perf_counter()))
        processing_time_ms = int((time.perf_counter() - started_at) * 1000)

        decision = self._business_decision_from_evidence(
            state,
            state.get("decision"),
        )
        approved_amount, rejected_amount = self._amounts_for_decision(
            state,
            decision,
        )

        payload = {
            "claim_id": state.get("claim", {}).get("claim_id", "unknown"),
            "decision": decision,
            "claim_status": self._claim_status_for_decision(decision),
            "approved_amount": approved_amount,
            "rejected_amount": rejected_amount,
            "currency": state.get("currency")
            or state.get("claim", {}).get("currency", ""),
            "policy_references": self._policy_references_for_decision(
                state,
                decision,
            ),
            "reason_codes": unique_preserve_order(state.get("reason_codes", [])),
            "missing_documents": unique_preserve_order(
                state.get("missing_documents", [])
            ),
            "confidence_score": float(state.get("confidence_score", 0.0)),
            "reviewer_required": decision == "Manual Review",
            "audit_trail": self._audit_summary(state),
            "processing_time_ms": processing_time_ms,
            "explanation": state.get("explanation", ""),
        }

        validated_output = ReimbursementDecision(**payload).model_dump()
        updated_state: AgentState = {
            **state,
            "claim_status": validated_output["claim_status"],
            "processing_time_ms": processing_time_ms,
            "final_output": validated_output,
            "tools_used": unique_preserve_order(
                list(state.get("tools_used", [])) + ["Output_Validator"]
            ),
        }

        return add_audit_event(
            updated_state,
            step="Output_Validator",
            status="success",
            message="Structured output validated.",
        )

    def route_after_tool_selection(self, state: AgentState) -> str:
        """Route to the first selected tool or confidence scoring."""

        return self._next_tool_node(state, current_tool=None)

    def route_after_receipt_checker(self, state: AgentState) -> str:
        """Route after Receipt_Checker."""

        return self._next_tool_node(state, current_tool="Receipt_Checker")

    def route_after_expense_limit_checker(self, state: AgentState) -> str:
        """Route after Expense_Limit_Checker."""

        return self._next_tool_node(state, current_tool="Expense_Limit_Checker")

    def route_after_approval_checker(self, state: AgentState) -> str:
        """Route after Approval_Checker."""

        return self._next_tool_node(state, current_tool="Approval_Checker")

    def route_after_duplicate_checker(self, state: AgentState) -> str:
        """Route after Duplicate_Checker."""

        return self._next_tool_node(state, current_tool="Duplicate_Checker")

    def route_after_confidence(self, state: AgentState) -> str:
        """Route to manual review output or final decision."""

        if self._must_manual_review(state):
            return "explanation"

        return "final_decision"

    def _build_default_llm(self) -> LLMClient | None:
        """Build Gemini client when API settings are available."""

        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is missing; using fallback reasoning.")
            return None

        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )

    def _invoke_tool_selection_llm(
        self,
        state: AgentState,
    ) -> Dict[str, Any]:
        """Invoke Gemini for tool selection."""

        if self.llm is None:
            return {}

        try:
            prompt = build_tool_selection_prompt()
            messages = prompt.format_messages(
                **build_tool_selection_inputs(state)
            )
            response = self.llm.invoke(messages)
            return extract_json_object(str(response.content))
        except Exception as exc:
            logger.warning("Tool selection LLM failed: %s", exc)
            return {}

    def _invoke_final_decision_llm(
        self,
        state: AgentState,
    ) -> Dict[str, Any]:
        """Invoke Gemini for final decision generation."""

        if self.llm is None:
            return {}

        try:
            prompt = build_decision_prompt()
            messages = prompt.format_messages(**build_decision_inputs(state))
            response = self.llm.invoke(messages)
            return extract_json_object(str(response.content))
        except Exception as exc:
            logger.warning("Final decision LLM failed: %s", exc)
            return {}

    def _record_tool_selection(
        self,
        state: AgentState,
        tools_to_call: List[str],
        selection: Dict[str, Any],
    ) -> AgentState:
        """Persist Gemini or fallback tool selection in state."""

        valid_tools = [
            tool
            for tool in unique_preserve_order(tools_to_call)
            if tool in AVAILABLE_AGENT_TOOLS
        ]

        updated_state: AgentState = {
            **state,
            "tools_to_call": valid_tools,
            "llm_tool_selection": selection,
        }

        if (
            selection.get("manual_review_required")
            and self._must_manual_review(updated_state)
        ):
            updated_state["reviewer_required"] = True
            updated_state["decision"] = "Manual Review"

        return add_audit_event(
            updated_state,
            step="Gemini_Tool_Decision",
            status="completed",
            message="Tool selection completed.",
            details={"tools_to_call": valid_tools},
        )

    def _fallback_tool_selection(self, state: AgentState) -> List[str]:
        """Select tools with deterministic rules when Gemini is unavailable."""

        tools = ["Receipt_Checker", "Expense_Limit_Checker"]
        claim = state.get("claim", {})
        description = str(claim.get("description", "")).lower()

        if (
            float(claim.get("amount", 0) or 0) >= 1000
            or claim.get("travel_type") == "international"
            or "business class" in description
        ):
            tools.append("Approval_Checker")

        if claim.get("receipt_id") or claim.get("invoice_number"):
            tools.append("Duplicate_Checker")

        return unique_preserve_order(tools)

    def _merge_tool_evidence(
        self,
        state: AgentState,
        result: Dict[str, Any],
    ) -> AgentState:
        """Merge reason codes, missing documents, and policy outcomes."""

        updated_state: AgentState = {
            **state,
            "reason_codes": unique_preserve_order(
                list(state.get("reason_codes", []))
                + result.get("reason_codes", [])
            ),
            "missing_documents": unique_preserve_order(
                list(state.get("missing_documents", []))
                + result.get("missing_documents", [])
            ),
            "policy_references": unique_preserve_order(
                list(state.get("policy_references", []))
                + result.get("policy_references", [])
            ),
        }

        if result.get("decision") == "Reject":
            updated_state["decision"] = "Reject"
            updated_state["reviewer_required"] = False
        elif result.get("status") == "manual_review":
            updated_state["decision"] = "Manual Review"
            updated_state["reviewer_required"] = True
        elif (
            result.get("decision") == "Partial Approval"
            and not state.get("reviewer_required")
            and state.get("decision") != "Manual Review"
        ):
            updated_state["decision"] = "Partial Approval"
            updated_state["approved_amount"] = float(
                result.get("approved_amount", 0.0)
            )
            updated_state["rejected_amount"] = float(
                result.get("rejected_amount", 0.0)
            )

        return updated_state

    def _fallback_final_decision(self, state: AgentState) -> Dict[str, Any]:
        """Build a deterministic final decision from tool evidence."""

        claim = state.get("claim", {})
        amount = float(claim.get("amount", 0.0) or 0.0)

        if self._has_tool_decision(state, "Reject"):
            decision = "Reject"
            approved_amount = 0.0
            rejected_amount = amount
            reviewer_required = False
        elif self._must_manual_review(state):
            decision = "Manual Review"
            approved_amount = 0.0
            rejected_amount = amount
            reviewer_required = True
        elif self._has_tool_decision(state, "Partial Approval"):
            limit_result = state.get("tool_results", {}).get(
                "Expense_Limit_Checker",
                {},
            )
            decision = "Partial Approval"
            approved_amount = float(limit_result.get("approved_amount", 0.0))
            rejected_amount = float(limit_result.get("rejected_amount", 0.0))
            reviewer_required = False
        else:
            decision = "Approve"
            approved_amount = amount
            rejected_amount = 0.0
            reviewer_required = False

        return {
            "decision": decision,
            "approved_amount": approved_amount,
            "rejected_amount": rejected_amount,
            "reviewer_required": reviewer_required,
            "confidence_score": state.get("confidence_score", 90.0),
        }

    def _apply_decision_payload(
        self,
        state: AgentState,
        payload: Dict[str, Any],
    ) -> AgentState:
        """Apply LLM or fallback decision payload to state."""

        decision = self._business_decision_from_evidence(
            state,
            payload.get("decision", state.get("decision")),
        )
        approved_amount, rejected_amount = self._amounts_for_decision(
            state,
            decision,
            payload,
        )

        return {
            **state,
            "decision": decision,
            "approved_amount": approved_amount,
            "rejected_amount": rejected_amount,
            "confidence_score": float(
                payload.get("confidence_score", state.get("confidence_score", 0))
            ),
            "reviewer_required": decision == "Manual Review",
            "reason_codes": unique_preserve_order(
                list(state.get("reason_codes", []))
                + payload.get("reason_codes", [])
            ),
            "missing_documents": unique_preserve_order(
                list(state.get("missing_documents", []))
                + payload.get("missing_documents", [])
            ),
            "policy_references": unique_preserve_order(
                list(state.get("policy_references", []))
                + payload.get("policy_references", [])
            )[:5],
        }

    def _next_tool_node(
        self,
        state: AgentState,
        current_tool: str | None,
    ) -> str:
        """Return the next selected tool node name."""

        selected_tools = state.get("tools_to_call", [])
        start_index = 0

        if current_tool is not None:
            start_index = TOOL_ORDER.index(current_tool) + 1

        node_names = {
            "Receipt_Checker": "receipt_checker",
            "Expense_Limit_Checker": "expense_limit_checker",
            "Approval_Checker": "approval_checker",
            "Duplicate_Checker": "duplicate_checker",
        }

        for tool_name in TOOL_ORDER[start_index:]:
            if tool_name in selected_tools:
                return node_names[tool_name]

        return "confidence_calculator"

    def _requires_manual_review_before_llm(self, state: AgentState) -> bool:
        """Return True when validation or retrieval has already failed."""

        return (
            not state.get("validation_result", {}).get("is_valid", False)
            or not state.get("policy_context")
        )

    def _must_manual_review(self, state: AgentState) -> bool:
        """Return True for mandatory manual review scenarios."""

        if not state.get("policy_context"):
            return True

        if not state.get("validation_result", {}).get("is_valid", True):
            return True

        if state.get("claim", {}).get("is_policy_exception"):
            return True

        if state.get("missing_documents"):
            return True

        if float(state.get("confidence_score", 100.0) or 0.0) < 70:
            return True

        for result in state.get("tool_results", {}).values():
            if result.get("status") == "manual_review":
                return True
            if result.get("decision") == "Manual Review":
                return True

        return self._explicit_manual_review_policy_matched(state)

    def _business_decision_from_evidence(
        self,
        state: AgentState,
        requested_decision: str | None = None,
    ) -> str:
        """Resolve the final decision from concrete tool and policy evidence."""

        if self._has_tool_decision(state, "Reject"):
            return "Reject"

        if self._must_manual_review(state):
            return "Manual Review"

        if self._has_tool_decision(state, "Partial Approval"):
            return "Partial Approval"

        if requested_decision in {"Approve", "Partial Approval", "Reject"}:
            return requested_decision

        return "Approve"

    def _amounts_for_decision(
        self,
        state: AgentState,
        decision: str,
        payload: Dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Return approved and rejected amounts for the resolved decision."""

        payload = payload or {}
        claim_amount = float(state.get("claim", {}).get("amount", 0.0) or 0.0)

        if decision == "Approve":
            return claim_amount, 0.0

        if decision == "Reject":
            return 0.0, claim_amount

        if decision == "Manual Review":
            return 0.0, claim_amount

        limit_result = state.get("tool_results", {}).get(
            "Expense_Limit_Checker",
            {},
        )

        approved_amount = float(
            payload.get(
                "approved_amount",
                limit_result.get("approved_amount", state.get("approved_amount", 0.0)),
            )
            or 0.0
        )
        rejected_amount = float(
            payload.get(
                "rejected_amount",
                limit_result.get("rejected_amount", state.get("rejected_amount", 0.0)),
            )
            or 0.0
        )

        if approved_amount == 0.0 and rejected_amount == 0.0:
            approved_amount = claim_amount

        return approved_amount, rejected_amount

    def _policy_references_for_decision(
        self,
        state: AgentState,
        decision: str,
    ) -> List[str]:
        """Return only policy IDs that support the final decision."""

        tool_results = state.get("tool_results", {})
        references: List[str] = []

        if decision == "Reject":
            duplicate_result = tool_results.get("Duplicate_Checker", {})
            if duplicate_result.get("decision") == "Reject":
                references.extend(duplicate_result.get("policy_references", []))

            limit_result = tool_results.get("Expense_Limit_Checker", {})
            if limit_result.get("decision") == "Reject":
                references.extend(limit_result.get("policy_references", []))
        elif decision == "Partial Approval":
            references.extend(
                self._tool_policy_references(tool_results, ["Expense_Limit_Checker"])
            )
        elif decision == "Manual Review":
            if state.get("claim", {}).get("is_policy_exception"):
                references.append("POL-MANUAL-003")

            for result in tool_results.values():
                if (
                    result.get("status") == "manual_review"
                    or result.get("decision") == "Manual Review"
                ):
                    references.extend(result.get("policy_references", []))

            references.extend(
                ref
                for ref in state.get("policy_references", [])
                if ref.startswith(("POL-MANUAL", "POL-RECEIPT", "POL-MGMT"))
            )
        else:
            references.extend(
                self._tool_policy_references(tool_results, ["Expense_Limit_Checker"])
            )

        if not references:
            references.extend(
                self._relevant_claim_policy_references(
                    state.get("policy_references", []),
                    state.get("claim", {}),
                    decision,
                )
            )

        return unique_preserve_order(references)[:5]

    @staticmethod
    def _tool_policy_references(
        tool_results: Dict[str, Any],
        tool_names: List[str],
    ) -> List[str]:
        """Collect policy references from selected tool outputs."""

        references: List[str] = []
        for tool_name in tool_names:
            result = tool_results.get(tool_name, {})
            references.extend(result.get("policy_references", []))

        return references

    def _relevant_claim_policy_references(
        self,
        references: List[str],
        claim: Dict[str, Any],
        decision: str,
    ) -> List[str]:
        """Filter retrieved policy references to the claim category."""

        expense_type = str(claim.get("expense_type", "")).lower()
        prefixes = {
            "hotel": ("POL-HOTEL", "POL-LIM"),
            "meal": ("POL-MEAL", "POL-LIM"),
            "laundry": ("POL-LAUND", "POL-LIM"),
            "flight": ("POL-FLIGHT",),
            "transport": ("POL-TRANS", "POL-LIM"),
        }.get(expense_type, ())

        if decision == "Manual Review":
            prefixes = prefixes + ("POL-MANUAL", "POL-RECEIPT", "POL-MGMT")

        return [ref for ref in references if ref.startswith(prefixes)]

    def _explicit_manual_review_policy_matched(self, state: AgentState) -> bool:
        """Return True only when claim facts match explicit review policies."""

        claim = state.get("claim", {})
        description = str(claim.get("description", "")).lower()

        if any(term in description for term in ["weekend", "saturday", "sunday"]):
            return True

        if any(term in description for term in ["luxury", "suite", "st. regis", "ritz"]):
            return True

        if "business class" in description and not claim.get("vp_approval"):
            return True

        if "late submission" in description or "over 90 days" in description:
            return True

        return False

    @staticmethod
    def _document_policy_references(documents: List[Any]) -> List[str]:
        """Read policy IDs from retrieved document metadata."""

        references = [
            str(doc.metadata.get("policy_id", ""))
            for doc in documents
            if str(doc.metadata.get("policy_id", "")) not in {"", "UNKNOWN"}
        ]

        if references:
            return unique_preserve_order(references)

        return unique_preserve_order(
            reference
            for doc in documents
            for reference in extract_policy_references(str(doc.page_content))
        )

    @staticmethod
    def _has_tool_decision(state: AgentState, decision: str) -> bool:
        """Return True when any tool recommends a decision."""

        return any(
            result.get("decision") == decision
            for result in state.get("tool_results", {}).values()
        )

    @staticmethod
    def _claim_status_for_decision(decision: str) -> str:
        """Map decision to final claim status."""

        return {
            "Approve": "Approved",
            "Partial Approval": "Partially Approved",
            "Reject": "Rejected",
            "Manual Review": "Manual Review",
        }.get(decision, "Manual Review")

    @staticmethod
    def _audit_summary(state: AgentState) -> Dict[str, Any]:
        """Build compact Appendix-N audit flags from graph state."""

        tools_used = set(state.get("tools_used", []))

        return {
            "policy_retrieved": "Policy_Retriever" in tools_used,
            "receipt_checked": "Receipt_Checker" in tools_used,
            "expense_limit_checked": "Expense_Limit_Checker" in tools_used,
            "duplicate_checked": "Duplicate_Checker" in tools_used,
            "approval_checked": "Approval_Checker" in tools_used,
            "confidence_calculated": "Confidence_Calculator" in tools_used,
            "explanation_generated": "Explanation_Generator" in tools_used,
            "validator_passed": state.get("validation_result", {}).get(
                "is_valid",
                False,
            ),
            "llm_reasoning_completed": bool(state.get("llm_tool_selection")),
        }
