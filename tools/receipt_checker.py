"""
Receipt completeness checker.

This tool validates whether the claim has enough receipt evidence for an
automated decision. It does not perform OCR; it checks submitted metadata and
routes missing evidence to Manual Review.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReceiptChecker:
    """Checks receipt evidence completeness for a claim."""

    def execute(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check whether required receipt details are present.

        Args:
            claim: Claim dictionary.

        Returns:
            Structured receipt check result.
        """

        missing_documents: List[str] = []
        reason_codes: List[str] = []

        if not claim.get("receipt_uploaded"):
            missing_documents.append("Itemized receipt")
            reason_codes.append("RC001")

        status = "manual_review" if missing_documents else "success"
        decision = "Manual Review" if missing_documents else "Approve"

        logger.info("Receipt check completed with status: %s", status)

        return {
            "tool_name": "Receipt_Checker",
            "status": status,
            "decision": decision,
            "missing_documents": missing_documents,
            "reason_codes": reason_codes,
            "policy_references": (
                ["POL-RECEIPT-003", "POL-MANUAL-001"]
                if missing_documents
                else []
            ),
            "reason": (
                "Receipt evidence is incomplete."
                if missing_documents
                else "Receipt evidence is present."
            ),
            "audit": {
                "receipt_uploaded": bool(claim.get("receipt_uploaded")),
                "invoice_number_present": bool(claim.get("invoice_number")),
                "receipt_id_present": bool(claim.get("receipt_id")),
            },
        }
