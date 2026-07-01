"""
Confidence calculator for reimbursement decisions.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """Calculates a transparent confidence score from graph evidence."""

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence from validation, policy retrieval,
        business tools, and final decision.

        Args:
            state: Current graph state.

        Returns:
            Structured confidence result.
        """

        confidence = 98.0
        reasons = []

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        validation_result = state.get("validation_result", {})

        if not validation_result.get("is_valid", False):
            confidence -= 35
            reasons.append("Claim validation failed.")

        # -----------------------------------------------------
        # Policy Retrieval
        # -----------------------------------------------------

        if not state.get("policy_context"):
            confidence -= 25
            reasons.append("Policy context could not be retrieved.")

        # -----------------------------------------------------
        # Receipt Check
        # -----------------------------------------------------

        claim = state.get("claim", {})

        if not claim.get("receipt_uploaded", False):
            confidence -= 20
            reasons.append("Receipt not uploaded.")

        # -----------------------------------------------------
        # Missing Documents
        # -----------------------------------------------------

        missing_documents = state.get("missing_documents", [])

        if missing_documents:
            deduction = min(len(missing_documents) * 5, 15)
            confidence -= deduction
            reasons.append(
                f"Missing documents: {', '.join(missing_documents)}."
            )

        # -----------------------------------------------------
        # Tool Results
        # -----------------------------------------------------

        for tool_name, result in state.get("tool_results", {}).items():

            status = result.get("status", "").lower()
            decision = result.get("decision", "")

            if status == "failed":
                confidence -= 15
                reasons.append(f"{tool_name} failed.")

            elif status == "manual_review":
                confidence -= 15
                reasons.append(f"{tool_name} requested manual review.")

            if decision == "Partial Approval":
                confidence -= 5
                reasons.append(
                    f"{tool_name} recommended partial approval."
                )

            elif decision == "Reject":
                confidence -= 10
                reasons.append(
                    f"{tool_name} recommended rejection."
                )

        # -----------------------------------------------------
        # Final Decision Adjustment
        # -----------------------------------------------------

        final_decision = state.get("decision", "")

        if final_decision == "Partial Approval":
            confidence -= 5

        elif final_decision == "Reject":
            confidence -= 10

        elif final_decision == "Manual Review":
            confidence -= 20

        # -----------------------------------------------------
        # Policy Exception
        # -----------------------------------------------------

        if claim.get("is_policy_exception", False):
            confidence -= 15
            reasons.append("Policy exception requested.")

        # -----------------------------------------------------
        # Bound Score
        # -----------------------------------------------------

        confidence = max(50.0, min(confidence, 98.0))

        status = (
            "manual_review"
            if confidence < 70
            else "success"
        )

        logger.info("Confidence calculated: %.2f%%", confidence)

        return {
            "tool_name": "Confidence_Calculator",
            "status": status,
            "confidence_score": float(round(confidence)),
            "decision": (   
                "Manual Review"
                if confidence < 70
                else final_decision or "Approve"
            ),
            "reason": (
                "; ".join(reasons)
                if reasons
                else "All validation checks, policy retrieval, and business rules passed successfully."
            ),
            "audit": {
                "score": round(confidence),
                "reason_count": len(reasons),
                "reasons": reasons,
            },
        }