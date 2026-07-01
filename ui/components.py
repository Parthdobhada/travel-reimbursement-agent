"""
Reusable Streamlit UI components.
"""

from datetime import date
from typing import Any, Dict

import streamlit as st


def render_claim_form() -> Dict[str, Any] | None:
    """
    Render the reimbursement claim intake form.

    Returns:
        Claim payload when submitted, otherwise None.
    """

    with st.form("claim_form"):

        st.subheader("Travel Reimbursement Claim")

        claim_id = st.text_input(
            "Claim ID",
            placeholder="e.g. CLM-1001"
        )

        employee_id = st.text_input(
            "Employee ID",
            placeholder="e.g. EMP001"
        )

        employee_role = st.selectbox(
            "Employee Role",
            [
                "Select Employee Role",
                "Software Engineer",
                "Senior Engineer",
                "Manager",
                "Director",
                "VP",
            ],
        )

        expense_type = st.selectbox(
            "Expense Type",
            [
                "Select Expense Type",
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
            ],
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=0.0,
            step=100.0
        )

        currency = st.selectbox(
            "Currency",
            ["INR", "USD"]
        )

        expense_date = st.date_input(
            "Expense Date",
            value=date.today()
        )

        vendor = st.text_input(
            "Vendor",
            placeholder="e.g. Marriott Bangalore"
        )

        invoice_number = st.text_input(
            "Invoice Number",
            placeholder="e.g. INV-10025"
        )

        receipt_uploaded = st.checkbox(
            "Receipt Uploaded"
        )

        travel_type = st.selectbox(
            "Travel Type",
            [
                "Select Travel Type",
                "domestic",
                "international",
            ],
        )

        st.markdown("### Required Approvals")

        manager_approval = st.checkbox(
            "Manager Approval Attached"
        )

        director_approval = st.checkbox(
            "Director Approval Attached"
        )

        vp_approval = st.checkbox(
            "VP Approval Attached"
        )

        is_policy_exception = st.checkbox(
            "Policy Exception Requested"
        )

        description = st.text_area(
            "Business Purpose / Description",
            placeholder="Describe why this expense was incurred..."
        )

        submitted = st.form_submit_button(
            "Evaluate Claim"
        )

    if not submitted:
        return None

    # ------------------------
    # Basic Validation
    # ------------------------

    if not claim_id.strip():
        st.error("Claim ID is required.")
        return None

    if not employee_id.strip():
        st.error("Employee ID is required.")
        return None

    if employee_role == "Select Employee Role":
        st.error("Please select Employee Role.")
        return None

    if expense_type == "Select Expense Type":
        st.error("Please select Expense Type.")
        return None

    if travel_type == "Select Travel Type":
        st.error("Please select Travel Type.")
        return None

    if amount <= 0:
        st.error("Amount must be greater than zero.")
        return None

    if not description.strip():
        st.error("Description is required.")
        return None

    return {
        "claim_id": claim_id.strip(),
        "employee_id": employee_id.strip(),
        "employee_role": employee_role,
        "expense_type": expense_type,
        "amount": amount,
        "currency": currency,
        "expense_date": expense_date.isoformat(),
        "description": description.strip(),
        "receipt_uploaded": receipt_uploaded,
        "vendor": vendor.strip() or None,
        "invoice_number": invoice_number.strip() or None,
        "travel_type": travel_type,
        "manager_approval": manager_approval,
        "director_approval": director_approval,
        "vp_approval": vp_approval,
        "is_policy_exception": is_policy_exception,
    }


def render_decision(decision: Dict[str, Any]) -> None:
    """
    Render the final structured decision.

    Args:
        decision: Final JSON output from the LangGraph agent.
    """

    st.divider()

    st.header("AI Decision")

    decision_value = decision.get("decision", "Manual Review")

    if decision_value == "Approve":
        st.success(f"✅ {decision_value}")

    elif decision_value == "Reject":
        st.error(f"❌ {decision_value}")

    elif decision_value == "Partial Approval":
        st.warning(f"⚠️ {decision_value}")

    else:
        st.info(f"🧑‍💼 {decision_value}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approved Amount",
            f"{decision.get('approved_amount',0):,.2f}"
        )

    with col2:
        st.metric(
            "Rejected Amount",
            f"{decision.get('rejected_amount',0):,.2f}"
        )

    st.subheader("Explanation")
    st.write(decision.get("explanation", "No explanation available."))

    st.subheader("Structured JSON Output")
    st.json(decision)