"""
Claim Validator

Validates an expense claim before it is processed by the AI agent.

Responsibilities:
- Check required fields
- Validate amount
- Validate currency
- Validate expense type
- Validate receipt flag
- Validate expense date
"""

import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class ClaimValidator:
    """
    Validates incoming expense claims.
    """

    ALLOWED_EXPENSE_TYPES = {
        "hotel",
        "flight",
        "meal",
        "transport",
        "laundry",
        "visa",
        "internet",
        "parking",
        "fuel",
        "miscellaneous",
    }

    ALLOWED_CURRENCIES = {
        "INR",
        "USD",
    }

    REQUIRED_FIELDS = [
        "employee_id",
        "expense_type",
        "amount",
        "currency",
        "expense_date",
        "description",
        "receipt_uploaded",
    ]

    def validate(self, claim: Dict) -> Dict:
        """
        Validate an expense claim.
        """

        errors: List[str] = []
        warnings: List[str] = []

        # -------------------------
        # Required fields
        # -------------------------

        for field in self.REQUIRED_FIELDS:

            if field not in claim or claim[field] in [None, ""]:

                errors.append(f"Missing required field: {field}")

        # Stop further validation if required fields are missing
        if errors:

            return {
                "status": "failed",
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "validated_claim": None,
            }

        # -------------------------
        # Amount
        # -------------------------

        try:

            amount = float(claim["amount"])

            if amount <= 0:

                errors.append("Amount must be greater than zero.")

        except Exception:

            errors.append("Amount must be numeric.")

        # -------------------------
        # Currency
        # -------------------------

        currency = claim["currency"].upper()

        if currency not in self.ALLOWED_CURRENCIES:

            errors.append(
                f"Unsupported currency: {currency}"
            )

        # -------------------------
        # Expense Type
        # -------------------------

        expense_type = claim["expense_type"].lower()

        if expense_type not in self.ALLOWED_EXPENSE_TYPES:

            warnings.append(
                f"Unknown expense type: {expense_type}"
            )

        # -------------------------
        # Date
        # -------------------------

        try:

            datetime.strptime(
                claim["expense_date"],
                "%Y-%m-%d",
            )

        except Exception:

            errors.append(
                "Expense date must be YYYY-MM-DD."
            )

        # -------------------------
        # Receipt Flag
        # -------------------------

        if not isinstance(
            claim["receipt_uploaded"],
            bool,
        ):

            errors.append(
                "receipt_uploaded must be True or False."
            )

        # -------------------------
        # Description
        # -------------------------

        if len(claim["description"].strip()) < 10:

            warnings.append(
                "Description is too short."
            )

        logger.info(
            "Claim validation completed."
        )

        return {

            "status": "success" if len(errors) == 0 else "failed",

            "is_valid": len(errors) == 0,

            "errors": errors,

            "warnings": warnings,

            "validated_claim": claim,

        }