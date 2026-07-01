"""
Structured output schema for reimbursement decisions.

This mirrors the assessment's Appendix N fields while keeping validation local
and testable.
"""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


Decision = Literal[
    "Approve",
    "Partial Approval",
    "Reject",
    "Manual Review",
]

ClaimStatus = Literal[
    "Pending",
    "Processing",
    "Approved",
    "Partially Approved",
    "Rejected",
    "Manual Review",
    "Completed",
]


class ReimbursementDecision(BaseModel):
    """Final JSON response returned by the LangGraph workflow."""

    claim_id: str
    decision: Decision
    claim_status: ClaimStatus
    approved_amount: float = Field(..., ge=0)
    rejected_amount: float = Field(..., ge=0)
    currency: str
    policy_references: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    confidence_score: float
    reviewer_required: bool
    audit_trail: Dict[str, Any]
    processing_time_ms: int = Field(..., ge=0)
    explanation: str

    @field_validator("confidence_score", mode="before")
    @classmethod
    def validate_confidence_score(cls, value: Any) -> float:
        """Validate and normalize confidence scores to a numeric percent."""

        if isinstance(value, str):
            value = value.strip()
            if value.endswith("%"):
                value = value.rstrip("%").strip()

        numeric_value = float(value)
        if numeric_value < 0 or numeric_value > 100:
            raise ValueError("confidence_score must be between 0 and 100")

        return numeric_value
