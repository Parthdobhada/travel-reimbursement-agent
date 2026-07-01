from agent.state import add_audit_event, add_tool_result, create_initial_state


def test_create_initial_state_has_safe_defaults():
    claim = {
        "claim_id": "CLM-001",
        "employee_id": "EMP001",
        "expense_type": "hotel",
        "amount": 7500,
        "currency": "INR",
        "expense_date": "2026-06-20",
        "description": "Hotel stay during client meeting in Bangalore.",
        "receipt_uploaded": True,
    }

    state = create_initial_state(claim)

    assert state["claim"] == claim
    assert state["claim_status"] == "Pending"
    assert state["currency"] == "INR"
    assert state["tool_results"] == {}
    assert state["tools_used"] == []
    assert state["audit_trail"][0]["step"] == "claim_intake"


def test_add_audit_event_appends_without_mutating_original_state():
    state = create_initial_state({"claim_id": "CLM-002"})

    updated_state = add_audit_event(
        state=state,
        step="validator",
        status="success",
        message="Claim validation passed.",
    )

    assert len(state["audit_trail"]) == 1
    assert len(updated_state["audit_trail"]) == 2
    assert updated_state["audit_trail"][-1]["step"] == "validator"


def test_add_tool_result_records_result_and_tool_usage_once():
    state = create_initial_state({"claim_id": "CLM-003"})

    first_update = add_tool_result(
        state=state,
        tool_name="Expense_Limit_Checker",
        result={
            "status": "success",
            "decision": "Approve",
            "approved_amount": 1000.0,
            "rejected_amount": 0.0,
            "reason": "Claim is within policy limit.",
        },
    )
    second_update = add_tool_result(
        state=first_update,
        tool_name="Expense_Limit_Checker",
        result={
            "status": "success",
            "decision": "Approve",
            "approved_amount": 1000.0,
            "rejected_amount": 0.0,
            "reason": "Claim is within policy limit.",
        },
    )

    assert "Expense_Limit_Checker" in second_update["tool_results"]
    assert second_update["tools_used"] == ["Expense_Limit_Checker"]
    assert second_update["audit_trail"][-1]["step"] == "Expense_Limit_Checker"
