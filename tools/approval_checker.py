"""
Approval requirement checker.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ApprovalChecker:
    """Checks whether the claim needs additional approval evidence."""

    HIGH_VALUE_INR_THRESHOLD = 50000
    HIGH_VALUE_USD_THRESHOLD = 1000

    def execute(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate manager, director, and VP approval needs.

        Args:
            claim: Claim dictionary.

        Returns:
            Structured approval check result.
        """

        missing_documents: List[str] = []
        reason_codes: List[str] = []
        required_approvals: List[str] = []

        amount = float(claim.get("amount", 0) or 0)
        currency = str(claim.get("currency", "")).upper()
        travel_type = str(claim.get("travel_type", "")).lower()
        description = str(claim.get("description", "")).lower()

        if self._is_high_value(amount, currency):
            required_approvals.append("Manager approval")
            if not claim.get("manager_approval"):
                missing_documents.append("Manager approval")
                reason_codes.append("RC027")

        if travel_type == "international":
            required_approvals.append("VP or Regional Director approval")
            if not (claim.get("vp_approval") or claim.get("director_approval")):
                missing_documents.append("VP or Regional Director approval")
                reason_codes.append("RC027")

        if "business class" in description and not claim.get("vp_approval"):
            required_approvals.append("VP approval for business class")
            missing_documents.append("VP approval for business class")
            reason_codes.append("RC005")

        status = "manual_review" if missing_documents else "success"

        logger.info("Approval check completed with status: %s", status)

        return {
            "tool_name": "Approval_Checker",
            "status": status,
            "decision": "Manual Review" if missing_documents else "Approve",
            "required_approvals": required_approvals,
            "missing_documents": missing_documents,
            "reason_codes": reason_codes,
            "policy_references": ["POL-MGMT-001"] if missing_documents else [],
            "reason": (
                "Required approval evidence is missing."
                if missing_documents
                else "Required approval evidence is present."
            ),
            "audit": {
                "amount": amount,
                "currency": currency,
                "travel_type": travel_type,
                "manager_approval": bool(claim.get("manager_approval")),
                "director_approval": bool(claim.get("director_approval")),
                "vp_approval": bool(claim.get("vp_approval")),
            },
        }

    def _is_high_value(self, amount: float, currency: str) -> bool:
        """Return True when the amount crosses approval thresholds."""

        if currency == "INR":
            return amount >= self.HIGH_VALUE_INR_THRESHOLD

        if currency == "USD":
            return amount >= self.HIGH_VALUE_USD_THRESHOLD

        return amount > 0
