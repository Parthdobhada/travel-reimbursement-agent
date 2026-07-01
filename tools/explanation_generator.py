"""
Explanation generator for final reimbursement responses.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Creates concise human-readable explanations from graph evidence."""

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an explanation grounded in decision, policy IDs, and tools.

        Args:
            state: Current graph state.

        Returns:
            Structured explanation result.
        """

        decision = state.get("decision", "Manual Review")
        claim = state.get("claim", {})
        expense_type = str(claim.get("expense_type", "expense")).lower()
        amount = float(claim.get("amount", 0) or 0)
        currency = claim.get("currency", "")
        approved_amount = float(state.get("approved_amount", 0) or 0)
        rejected_amount = float(state.get("rejected_amount", 0) or 0)
        policy_references = state.get("policy_references", [])
        reason_codes = state.get("reason_codes", [])
        missing_documents = state.get("missing_documents", [])
        receipt_uploaded = bool(claim.get("receipt_uploaded"))

        policy_text = (
            f" according to {', '.join(policy_references)}"
            if policy_references
            else ""
        )
        amount_text = self._money(amount, currency)
        approved_text = self._money(approved_amount, currency)
        rejected_text = self._money(rejected_amount, currency)

        if decision == "Approve":
            explanation = (
                f"The {expense_type} expense of {amount_text} is within the "
                "allowed reimbursement limit. "
                f"{self._receipt_sentence(receipt_uploaded)} "
                "No duplicate, missing approval, or policy exception was found. "
                f"Therefore the claim is approved{policy_text}."
            )
        elif decision == "Partial Approval":
            explanation = (
                f"The {expense_type} expense of {amount_text} exceeds the "
                "applicable reimbursement limit. "
                f"{approved_text} is approved and {rejected_text} is rejected"
                f"{policy_text}."
            )
        elif decision == "Reject":
            reason = self._primary_rejection_reason(state)
            explanation = (
                f"The {expense_type} expense of {amount_text} is rejected "
                f"because {reason}. The rejected amount is {rejected_text}"
                f"{policy_text}."
            )
        else:
            reason = self._manual_review_reason(state)
            explanation = (
                f"The {expense_type} expense of {amount_text} requires manual "
                f"review because {reason}. The pending amount is {rejected_text}"
                f"{policy_text}."
            )

        if missing_documents:
            explanation += f" Missing documents: {', '.join(missing_documents)}."

        if reason_codes:
            explanation += f" Reason codes: {', '.join(reason_codes)}."

        explanation = f"{decision}: {explanation}"

        logger.info("Generated explanation for decision: %s", decision)

        return {
            "tool_name": "Explanation_Generator",
            "status": "success",
            "explanation": explanation,
            "reason": "Explanation generated from final graph evidence.",
            "audit": {
                "policy_reference_count": len(policy_references),
                "reason_code_count": len(reason_codes),
                "missing_document_count": len(missing_documents),
            },
        }

    @staticmethod
    def _money(amount: float, currency: str) -> str:
        """Format money without introducing locale dependencies."""

        return f"{currency} {amount:,.2f}".strip()

    @staticmethod
    def _receipt_sentence(receipt_uploaded: bool) -> str:
        """Return a receipt evidence sentence."""

        if receipt_uploaded:
            return "A receipt was provided."

        return "A receipt was not provided."

    @staticmethod
    def _primary_rejection_reason(state: Dict[str, Any]) -> str:
        """Describe the strongest rejection reason."""

        tool_results = state.get("tool_results", {})

        duplicate_result = tool_results.get("Duplicate_Checker", {})
        if duplicate_result.get("decision") == "Reject":
            return duplicate_result.get("reason", "a duplicate claim was detected")

        limit_result = tool_results.get("Expense_Limit_Checker", {})
        if limit_result.get("decision") == "Reject":
            return limit_result.get("reason", "the expense is not reimbursable")

        return "it violates the reimbursement policy"

    @staticmethod
    def _manual_review_reason(state: Dict[str, Any]) -> str:
        """Describe why a claim needs manual review."""

        if state.get("missing_documents"):
            return "required evidence is missing"

        if state.get("claim", {}).get("is_policy_exception"):
            return "a policy exception was requested"

        if not state.get("policy_context"):
            return "no relevant policy context was retrieved"

        if float(state.get("confidence_score", 100.0) or 0.0) < 70:
            return "the decision confidence is below the auto-decision threshold"

        for result in state.get("tool_results", {}).values():
            if result.get("status") == "manual_review":
                return result.get("reason", "a tool requested manual review")

        return "an explicit manual-review policy matched the claim facts"
