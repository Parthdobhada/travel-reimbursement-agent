"""
Duplicate claim checker.

The tool compares a submitted claim against supplied historical claim records.
In production this can be backed by a database; for this project it accepts the
history as structured data so the LangGraph node remains testable.
"""

import logging
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


class DuplicateChecker:
    """Detects likely duplicate reimbursement claims."""

    def execute(
        self,
        claim: Dict[str, Any],
        historical_claims: Iterable[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Check whether the claim appears to duplicate previous submissions.

        Args:
            claim: Current claim dictionary.
            historical_claims: Optional previous claims.

        Returns:
            Structured duplicate check result.
        """

        history = list(historical_claims or [])
        matches = []

        for historical_claim in history:
            if self._is_duplicate(claim, historical_claim):
                matches.append(historical_claim.get("claim_id", "unknown"))

        is_duplicate = len(matches) > 0
        status = "failed" if is_duplicate else "success"

        logger.info("Duplicate check completed. duplicate=%s", is_duplicate)

        return {
            "tool_name": "Duplicate_Checker",
            "status": status,
            "decision": "Reject" if is_duplicate else "Approve",
            "reason_codes": ["RC002"] if is_duplicate else [],
            "policy_references": ["POL-DUP-001"] if is_duplicate else [],
            "duplicate_claim_ids": matches,
            "reason": (
                "Potential duplicate claim detected."
                if is_duplicate
                else "No duplicate claim detected."
            ),
            "audit": {
                "historical_claims_checked": len(history),
                "duplicate_matches": matches,
            },
        }

    @staticmethod
    def _is_duplicate(
        claim: Dict[str, Any],
        historical_claim: Dict[str, Any],
    ) -> bool:
        """Return True when identifiers or key transaction fields match."""

        receipt_id = claim.get("receipt_id")
        invoice_number = claim.get("invoice_number")

        if receipt_id and receipt_id == historical_claim.get("receipt_id"):
            return True

        if (
            invoice_number
            and invoice_number == historical_claim.get("invoice_number")
        ):
            return True

        comparable_fields = [
            "employee_id",
            "vendor",
            "amount",
            "currency",
            "expense_date",
        ]

        return all(
            claim.get(field) is not None
            and claim.get(field) == historical_claim.get(field)
            for field in comparable_fields
        )
