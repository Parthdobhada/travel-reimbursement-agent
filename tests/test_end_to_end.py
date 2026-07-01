from tests.test_graph import build_test_agent


def test_end_to_end_duplicate_claim_is_rejected():
    agent = build_test_agent(
        policy_context="POL-TRANS-001 Standard taxi rides are reimbursable up to $100.",
        history=[
            {
                "claim_id": "OLD-DUP",
                "employee_id": "EMP001",
                "vendor": "Uber",
                "amount": 40,
                "currency": "USD",
                "expense_date": "2026-06-21",
                "invoice_number": "INV-DUP-1",
            }
        ],
    )

    result = agent.process_claim(
        {
            "claim_id": "CLM-DUP",
            "employee_id": "EMP001",
            "expense_type": "transport",
            "amount": 40,
            "currency": "USD",
            "expense_date": "2026-06-21",
            "description": "Uber ride from airport to hotel.",
            "receipt_uploaded": True,
            "vendor": "Uber",
            "invoice_number": "INV-DUP-1",
        }
    )

    assert result["decision"] == "Reject"
    assert result["rejected_amount"] == 40
    assert "RC002" in result["reason_codes"]


def test_end_to_end_missing_mandatory_field_goes_to_manual_review():
    agent = build_test_agent()

    result = agent.process_claim(
        {
            "claim_id": "CLM-BAD",
            "employee_id": "EMP001",
            "expense_type": "meal",
            "amount": 40,
            "currency": "USD",
            "description": "Lunch during client meeting in London.",
            "receipt_uploaded": True,
        }
    )

    assert result["decision"] == "Manual Review"
    assert result["reviewer_required"] is True
    assert any("expense_date" in item for item in result["missing_documents"])
