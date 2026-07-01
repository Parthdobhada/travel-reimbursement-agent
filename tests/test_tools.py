from tools.approval_checker import ApprovalChecker
from tools.confidence_calculator import ConfidenceCalculator
from tools.duplicate_checker import DuplicateChecker
from tools.explanation_generator import ExplanationGenerator
from tools.receipt_checker import ReceiptChecker


def test_receipt_checker_routes_missing_receipt_to_manual_review():
    result = ReceiptChecker().execute(
        {
            "claim_id": "CLM-MISS",
            "amount": 75,
            "receipt_uploaded": False,
        }
    )

    assert result["status"] == "manual_review"
    assert result["decision"] == "Manual Review"
    assert "RC001" in result["reason_codes"]
    assert result["missing_documents"] == ["Itemized receipt"]


def test_duplicate_checker_rejects_matching_invoice_number():
    result = DuplicateChecker().execute(
        {"invoice_number": "INV-1", "amount": 100},
        historical_claims=[{"claim_id": "OLD-1", "invoice_number": "INV-1"}],
    )

    assert result["decision"] == "Reject"
    assert result["duplicate_claim_ids"] == ["OLD-1"]
    assert "RC002" in result["reason_codes"]


def test_approval_checker_requests_vp_approval_for_business_class():
    result = ApprovalChecker().execute(
        {
            "amount": 900,
            "currency": "USD",
            "description": "Business class flight from Mumbai to Dubai.",
            "travel_type": "international",
            "vp_approval": False,
        }
    )

    assert result["decision"] == "Manual Review"
    assert "VP approval for business class" in result["missing_documents"]
    assert "RC005" in result["reason_codes"]


def test_confidence_calculator_drops_score_for_missing_policy():
    result = ConfidenceCalculator().execute(
        {
            "validation_result": {"is_valid": True},
            "policy_context": "",
            "tool_results": {},
            "missing_documents": [],
        }
    )

    assert result["status"] == "manual_review"
    assert result["confidence_score"] < 70


def test_explanation_generator_uses_policy_and_reason_codes():
    result = ExplanationGenerator().execute(
        {
            "decision": "Partial Approval",
            "policy_references": ["POL-MEAL-001"],
            "reason_codes": ["RC004"],
            "missing_documents": [],
        }
    )

    assert "Partial Approval" in result["explanation"]
    assert "POL-MEAL-001" in result["explanation"]
    assert "RC004" in result["explanation"]
