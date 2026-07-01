from agent.prompts import (
    AVAILABLE_AGENT_TOOLS,
    APPENDIX_N_OUTPUT_SCHEMA,
    build_decision_inputs,
    build_decision_prompt,
    build_tool_selection_inputs,
    build_tool_selection_prompt,
)
from agent.state import add_tool_result, create_initial_state


def test_tool_selection_prompt_formats_required_context():
    state = create_initial_state(
        {
            "claim_id": "CLM-101",
            "expense_type": "hotel",
            "amount": 9000,
            "currency": "INR",
            "receipt_uploaded": True,
        }
    )
    state["validation_result"] = {"is_valid": True}
    state["policy_context"] = "POL-HOTEL-001 Max INR 8000 per night."

    messages = build_tool_selection_prompt().format_messages(
        **build_tool_selection_inputs(state)
    )
    prompt_text = "\n".join(message.content for message in messages)

    assert "CLM-101" in prompt_text
    assert "POL-HOTEL-001" in prompt_text
    assert "Expense_Limit_Checker" in prompt_text
    assert "Manual Review" in prompt_text


def test_decision_prompt_formats_appendix_n_schema_and_tool_results():
    state = create_initial_state(
        {
            "claim_id": "CLM-102",
            "expense_type": "meal",
            "amount": 95,
            "currency": "USD",
            "receipt_uploaded": True,
        }
    )
    state["validation_result"] = {"is_valid": True}
    state["policy_context"] = "POL-MEAL-001 Dinner limit is $75."
    state = add_tool_result(
        state=state,
        tool_name="Expense_Limit_Checker",
        result={
            "status": "success",
            "decision": "Partial Approval",
            "approved_amount": 75.0,
            "rejected_amount": 20.0,
        },
    )

    messages = build_decision_prompt().format_messages(
        **build_decision_inputs(state)
    )
    prompt_text = "\n".join(message.content for message in messages)

    assert "CLM-102" in prompt_text
    assert "POL-MEAL-001" in prompt_text
    assert "Partial Approval" in prompt_text
    assert "retrieval_confidence" in prompt_text
    assert "audit_trail" in prompt_text


def test_prompt_constants_include_required_tools_and_output_fields():
    assert "Receipt_Checker" in AVAILABLE_AGENT_TOOLS
    assert "Expense_Limit_Checker" in AVAILABLE_AGENT_TOOLS
    assert "Approval_Checker" in AVAILABLE_AGENT_TOOLS
    assert APPENDIX_N_OUTPUT_SCHEMA["decision"]
    assert APPENDIX_N_OUTPUT_SCHEMA["missing_documents"]
