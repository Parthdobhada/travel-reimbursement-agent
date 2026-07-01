"""
Streamlit app for the Enterprise AI Travel Reimbursement Agent.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agent.graph import TravelReimbursementAgent
from ui.components import render_claim_form, render_decision
from utils.logger import setup_logging


@st.cache_resource
def get_agent(history: List[Dict[str, Any]]) -> TravelReimbursementAgent:
    """
    Create and cache the LangGraph agent.

    Args:
        history: Historical claims used for duplicate checks.

    Returns:
        TravelReimbursementAgent instance.
    """

    setup_logging()
    return TravelReimbursementAgent(historical_claims=history)


def load_sample_history() -> List[Dict[str, Any]]:
    """
    Load sample duplicate-check history.

    Returns:
        Historical claim list.
    """

    path = Path(__file__).resolve().parent.parent / "sample_data" / "sample_claims.json"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, list) else []


def main() -> None:
    """Render the Streamlit app."""

    st.set_page_config(
        page_title="Travel Reimbursement Agent",
        page_icon="",
        layout="wide",
    )

    st.title("Enterprise AI Travel Reimbursement Agent")

    history = load_sample_history()
    agent = get_agent(history)
    claim = render_claim_form()

    if claim:
        with st.spinner("Evaluating claim through LangGraph..."):
            decision = agent.process_claim(claim)

        render_decision(decision)


if __name__ == "__main__":
    main()
