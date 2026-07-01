"""
Reusable Streamlit UI components — Travel & Expense theme.
"""

from datetime import date
from typing import Any, Dict

import streamlit as st

# ----------------------------------------------------------------------
# Icons / visual metadata
# ----------------------------------------------------------------------

EXPENSE_ICONS = {
    "hotel": "🏨",
    "flight": "✈️",
    "meal": "🍽️",
    "transport": "🚕",
    "laundry": "🧺",
    "visa": "🛂",
    "internet": "📶",
    "parking": "🅿️",
    "fuel": "⛽",
    "miscellaneous": "📎",
}

ROLE_ICONS = {
    "Software Engineer": "💻",
    "Senior Engineer": "🧑‍💻",
    "Manager": "🧑‍💼",
    "Director": "🎯",
    "VP": "🏆",
}


def inject_theme() -> None:
    """Inject custom CSS to give the app a fresh, travel-inspired look."""

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6f9ff 0%, #ffffff 220px);
        }

        .travel-hero {
            background: linear-gradient(120deg, #0f2027 0%, #2c5364 55%, #4facfe 100%);
            border-radius: 18px;
            padding: 2.2rem 2rem;
            margin-bottom: 1.6rem;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(15, 32, 39, 0.25);
            position: relative;
            overflow: hidden;
        }
        .travel-hero::after {
            content: "✈️  🧳  🗺️  🏨  🚕";
            position: absolute;
            right: -10px;
            bottom: -6px;
            font-size: 2.6rem;
            opacity: 0.18;
            letter-spacing: 6px;
        }
        .travel-hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
        }
        .travel-hero p {
            margin: 0.4rem 0 0 0;
            opacity: 0.9;
            font-size: 0.98rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #edf1f7;
            border-radius: 14px;
            padding: 1.1rem 1.3rem 0.4rem 1.3rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 14px rgba(30, 41, 59, 0.05);
        }
        .section-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #1f2d3d;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 2.2rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            box-shadow: 0 6px 16px rgba(229, 46, 113, 0.35);
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        .decision-badge {
            display: inline-block;
            padding: 0.4rem 1.1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.8rem;
        }
        .badge-approve { background: #e6f7ee; color: #0f9d58; }
        .badge-reject { background: #fdeceb; color: #d93025; }
        .badge-partial { background: #fff6e5; color: #b9770e; }
        .badge-manual { background: #eef2ff; color: #3949ab; }

        [data-testid="stMetric"] {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
            border: 1px solid #eef1f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render a colorful travel-themed hero banner at the top of the app."""

    st.markdown(
        """
        <div class="travel-hero">
            <h1>🧳 Travel & Expense Reimbursement</h1>
            <p>Submit your claim below — our AI agent reviews policy, approvals,
            and amounts in seconds so you get paid faster. ✈️ 🌍 🧾</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_form() -> Dict[str, Any] | None:
    """
    Render the reimbursement claim intake form.

    Returns:
        Claim payload when submitted, otherwise None.
    """

    inject_theme()
    render_hero()

    with st.form("claim_form"):

        # ---------------- Claim identity ----------------
        st.markdown(
            '<div class="section-card"><div class="section-title">🪪 Claim Details</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            claim_id = st.text_input("Claim ID", placeholder="e.g. CLM-1001")
        with c2:
            employee_id = st.text_input("Employee ID", placeholder="e.g. EMP001")

        c3, c4 = st.columns(2)
        with c3:
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
                format_func=lambda r: f"{ROLE_ICONS.get(r, '')} {r}".strip(),
            )
        with c4:
            travel_type = st.selectbox(
                "Travel Type",
                ["Select Travel Type", "domestic", "international"],
                format_func=lambda t: {
                    "domestic": "🏠 domestic",
                    "international": "🌍 international",
                }.get(t, t),
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Expense info ----------------
        st.markdown(
            '<div class="section-card"><div class="section-title">💳 Expense Info</div>',
            unsafe_allow_html=True,
        )
        c5, c6, c7 = st.columns(3)
        with c5:
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
                format_func=lambda e: f"{EXPENSE_ICONS.get(e, '')} {e}".strip(),
            )
        with c6:
            amount = st.number_input("Amount", min_value=0.0, value=0.0, step=100.0)
        with c7:
            currency = st.selectbox("Currency", ["INR", "USD"])

        c8, c9 = st.columns(2)
        with c8:
            expense_date = st.date_input("Expense Date", value=date.today())
        with c9:
            vendor = st.text_input("Vendor", placeholder="e.g. Marriott Bangalore")

        c10, c11 = st.columns(2)
        with c10:
            invoice_number = st.text_input("Invoice Number", placeholder="e.g. INV-10025")
        with c11:
            receipt_uploaded = st.checkbox("📎 Receipt Uploaded")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Approvals ----------------
        st.markdown(
            '<div class="section-card"><div class="section-title">✅ Required Approvals</div>',
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            manager_approval = st.checkbox("👤 Manager")
        with a2:
            director_approval = st.checkbox("🎯 Director")
        with a3:
            vp_approval = st.checkbox("🏆 VP")
        with a4:
            is_policy_exception = st.checkbox("⚠️ Exception")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Description ----------------
        st.markdown(
            '<div class="section-card"><div class="section-title">📝 Business Purpose</div>',
            unsafe_allow_html=True,
        )
        description = st.text_area(
            "Business Purpose / Description",
            placeholder="Describe why this expense was incurred...",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀 Evaluate Claim")

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
    st.markdown("## 🧾 AI Decision")

    decision_value = decision.get("decision", "Manual Review")

    badge_map = {
        "Approve": ("badge-approve", "✅"),
        "Reject": ("badge-reject", "❌"),
        "Partial Approval": ("badge-partial", "⚠️"),
    }
    badge_class, emoji = badge_map.get(decision_value, ("badge-manual", "🧑‍💼"))

    st.markdown(
        f'<span class="decision-badge {badge_class}">{emoji} {decision_value}</span>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Approved Amount", f"{decision.get('approved_amount', 0):,.2f}")

    with col2:
        st.metric("🚫 Rejected Amount", f"{decision.get('rejected_amount', 0):,.2f}")

    st.markdown("#### 📋 Explanation")
    st.write(decision.get("explanation", "No explanation available."))

    with st.expander("🔍 Structured JSON Output"):
        st.json(decision)