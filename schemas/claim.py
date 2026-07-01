"""
Pydantic schema for claim intake.

The Streamlit UI, tests, and LangGraph entrypoint can all use this schema to
normalize JSON, CSV, form, or API payloads before graph execution.
"""

from datetime import date
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Claim(BaseModel):
    """Travel reimbursement claim submitted by an employee."""

    model_config = ConfigDict(extra="allow")

    claim_id: str = Field(..., min_length=1)
    employee_id: str = Field(..., min_length=1)
    expense_type: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    expense_date: date
    description: str = Field(..., min_length=10)
    receipt_uploaded: bool
    vendor: Optional[str] = None
    receipt_id: Optional[str] = None
    invoice_number: Optional[str] = None
    employee_role: Optional[str] = None
    travel_type: Optional[str] = None
    manager_approval: bool = False
    director_approval: bool = False
    vp_approval: bool = False
    is_policy_exception: bool = False

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize currency codes to uppercase ISO-style values."""

        return value.upper()

    @field_validator("expense_type")
    @classmethod
    def normalize_expense_type(cls, value: str) -> str:
        """Normalize expense type values for downstream tools."""

        return value.lower()

    def to_claim_dict(self) -> Dict[str, Any]:
        """
        Convert the validated model into a plain dictionary for existing tools.

        Returns:
            Dictionary compatible with the existing ClaimValidator.
        """

        return self.model_dump(mode="json")
