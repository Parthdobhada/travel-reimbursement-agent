"""
Expense Limit Checker

Checks whether a claim amount exceeds the policy limits.
"""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ExpenseLimitChecker:
    """
    Checks claim amount against retrieved policy.
    """

    def execute(
        self,
        claim: Dict,
        policy_context: str,
    ) -> Dict:

        amount = float(claim["amount"])
        expense_type = str(claim.get("expense_type", "")).lower()

        if expense_type == "laundry":
            duration_days = self._trip_duration_days(claim)
            if duration_days is not None and duration_days < 5:
                return {
                    "status": "failed",
                    "decision": "Reject",
                    "approved_amount": 0,
                    "rejected_amount": amount,
                    "policy_limit": None,
                    "reason_codes": ["RC022"],
                    "policy_references": ["POL-LAUND-001"],
                    "reason": (
                        "Laundry is reimbursable only for trips exceeding "
                        "5 consecutive business days."
                    ),
                }

        policy_limit = self._category_policy_limit(claim)
        if policy_limit is None:
            policy_limit = self._fallback_policy_limit(policy_context)

        if policy_limit is None:
            return {
                "status": "manual_review",
                "decision": "Manual Review",
                "approved_amount": 0,
                "rejected_amount": amount,
                "reason": "No monetary limit found in policy.",
                "policy_limit": None,
                "policy_references": [],
            }

        return self._decision_for_limit(
            amount=amount,
            policy_limit=policy_limit,
            expense_type=expense_type,
        )

    def _decision_for_limit(
        self,
        amount: float,
        policy_limit: float,
        expense_type: str,
    ) -> Dict[str, Any]:
        """Return the reimbursement decision for a category limit."""

        if amount <= policy_limit:
            return {
                "status": "success",
                "decision": "Approve",
                "approved_amount": amount,
                "rejected_amount": 0,
                "policy_limit": policy_limit,
                "policy_references": self._policy_references(
                    expense_type,
                    exceeded_limit=False,
                ),
                "reason": "Claim is within policy limit.",
            }

        return {
            "status": "success",
            "decision": "Partial Approval",
            "approved_amount": policy_limit,
            "rejected_amount": round(amount - policy_limit, 2),
            "policy_limit": policy_limit,
            "reason_codes": self._limit_reason_codes(expense_type),
            "policy_references": self._policy_references(
                expense_type,
                exceeded_limit=True,
            ),
            "reason": "Claim exceeds policy limit.",
        }

    def _category_policy_limit(self, claim: Dict[str, Any]) -> float | None:
        """Return the applicable policy limit for known expense categories."""

        expense_type = str(claim.get("expense_type", "")).lower()
        currency = str(claim.get("currency", "")).upper()
        travel_type = str(claim.get("travel_type", "")).lower()
        description = str(claim.get("description", "")).lower()

        if expense_type == "hotel":
            if currency == "INR" or travel_type == "domestic":
                if any(
                    city in description
                    for city in ["pune", "chennai", "kolkata", "ahmedabad"]
                ):
                    return 6000.0
                if "tier 3" in description:
                    return 4500.0
                return 8000.0

            if any(
                city in description
                for city in ["berlin", "madrid", "sydney", "dubai"]
            ):
                return 250.0
            if "tier 3" in description:
                return 175.0
            return 350.0

        if expense_type == "meal":
            if currency == "INR" or travel_type == "domestic":
                if "breakfast" in description:
                    return 500.0
                if "lunch" in description:
                    return 800.0
                return 1200.0

            if any(
                term in description
                for term in ["client", "business meal", "guest"]
            ):
                return 150.0
            if "breakfast" in description:
                return 25.0
            if "lunch" in description:
                return 40.0
            return 75.0

        if expense_type == "laundry":
            return 30.0

        return None

    def _fallback_policy_limit(self, policy_context: str) -> float | None:
        """Extract a fallback limit if the category is not yet modelled."""

        amounts = []
        patterns = [
            r"₹\s?([\d,]+)",
            r"â‚¹\s?([\d,]+)",
            r"\$\s?([\d,]+)",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, policy_context):
                amounts.append(float(match.replace(",", "")))

        return max(amounts) if amounts else None

    def _trip_duration_days(self, claim: Dict[str, Any]) -> int | None:
        """Find trip duration from common structured fields or description."""

        for field in [
            "trip_duration_days",
            "trip_days",
            "travel_days",
            "duration_days",
            "number_of_days",
        ]:
            value = claim.get(field)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        description = str(claim.get("description", "")).lower()
        match = re.search(r"(\d+)\s*[- ]?\s*day", description)
        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def _limit_reason_codes(expense_type: str) -> list[str]:
        """Return category-specific limit reason codes."""

        return {
            "hotel": ["RC003"],
            "meal": ["RC004"],
        }.get(expense_type, [])

    @staticmethod
    def _policy_references(
        expense_type: str,
        exceeded_limit: bool,
    ) -> list[str]:
        """Return policy IDs used by the category limit check."""

        references = {
            "hotel": ["POL-HOTEL-001A"],
            "meal": ["POL-MEAL-001"],
            "laundry": ["POL-LAUND-001"],
        }.get(expense_type, ["POL-LIM-001"])

        if exceeded_limit and "POL-LIM-001" not in references:
            references.append("POL-LIM-001")

        return references
