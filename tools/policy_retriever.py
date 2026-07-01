"""
Policy Retriever Tool

Provides policy retrieval functionality for the AI Travel
Reimbursement Agent using the RAG pipeline.
"""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from rag.retriever import PolicyRetriever

logger = logging.getLogger(__name__)


class PolicyRetrieverTool:
    """
    Tool responsible for retrieving relevant policy information
    from the RAG knowledge base.
    """

    def __init__(self):
        self.retriever = PolicyRetriever()

    def execute(
        self,
        claim: Dict,
    ) -> Dict:
        """
        Retrieve policy context for a claim.

        Args:
            claim: Claim dictionary.

        Returns:
            Dictionary containing retrieved policy information.
        """

        expense_type = str(claim.get("expense_type", ""))
        travel_type = str(claim.get("travel_type", ""))
        amount = claim.get("amount", "")
        receipt_uploaded = claim.get("receipt_uploaded", False)
        description = str(claim.get("description", ""))
        trigger_terms = self._manual_review_trigger_terms(claim)
        policy_hints = self._primary_policy_hints(expense_type)

        query = f"""
        Expense Type: {expense_type}
        Travel Type: {travel_type}
        Amount: {amount}
        Receipt Uploaded: {receipt_uploaded}

        Description: {description}

        Retrieve the primary {expense_type} reimbursement policy,
        category spending limits, and standard reimbursement conditions.
        Prefer these policy IDs when relevant: {policy_hints}.
        Include manual review, exception, rejection, weekend, luxury, or
        missing-document policies only if these claim facts mention them:
        {", ".join(trigger_terms) if trigger_terms else "none"}.
        """

        logger.info(
            f"Retrieving policy for {expense_type}"
        )

        documents = self.retriever.retrieve(query, k=10)

        filtered_documents = [
            doc for doc in documents if self._is_relevant_document(doc, claim)
        ]

        # Fallback if filtering removes everything
        if not filtered_documents:
            filtered_documents = self._fallback_documents(documents, claim)

        primary_document = self._primary_policy_document(claim)
        if primary_document and not self._contains_primary_policy(
            filtered_documents,
            claim,
        ):
            filtered_documents.insert(0, primary_document)

        context = self.retriever.format_context(
            filtered_documents[:5]
        )

        return {
            "query": query,
            "policy_context": context,
            "retrieved_documents": filtered_documents[:5],
            "retrieval_status": "success"
        }

    def _is_relevant_document(self, doc: Any, claim: Dict) -> bool:
        """Return True when a retrieved policy matches the submitted claim."""

        expense_type = str(claim.get("expense_type", "")).lower()
        policy_id = str(doc.metadata.get("policy_id", "")).upper()
        policy_type = str(doc.metadata.get("policy_type", "")).lower()
        content = str(doc.page_content).lower()
        trigger_terms = self._manual_review_trigger_terms(claim)

        primary_prefixes = {
            "hotel": ["POL-HOTEL"],
            "meal": ["POL-MEAL"],
            "laundry": ["POL-LAUND"],
            "flight": ["POL-FLIGHT"],
            "transport": ["POL-TRANS"],
            "parking": ["POL-PARK"],
            "internet": ["POL-COMM"],
            "fuel": ["POL-VEH"],
            "visa": ["POL-VISA"],
        }.get(expense_type, [])

        if any(policy_id.startswith(prefix) for prefix in primary_prefixes):
            return True

        if policy_type == expense_type:
            return True

        if policy_id.startswith("POL-LIM") and expense_type in {
            "hotel",
            "meal",
            "laundry",
            "transport",
            "parking",
        }:
            return True

        if self._document_matches_trigger(policy_id, content, trigger_terms):
            return True

        if policy_id == "UNKNOWN" and expense_type and expense_type in content:
            return not self._contains_unmatched_escalation_policy(
                content,
                trigger_terms,
            )

        return False

    def _fallback_documents(self, documents: List[Any], claim: Dict) -> List[Any]:
        """Prefer documents that at least mention the expense category."""

        expense_type = str(claim.get("expense_type", "")).lower()
        trigger_terms = self._manual_review_trigger_terms(claim)
        safe_documents = [
            doc
            for doc in documents
            if not self._contains_unmatched_escalation_policy(
                str(doc.page_content).lower(),
                trigger_terms,
            )
        ]
        category_matches = [
            doc
            for doc in safe_documents
            if expense_type in str(doc.page_content).lower()
        ]

        return category_matches or safe_documents or documents[:1]

    def _document_matches_trigger(
        self,
        policy_id: str,
        content: str,
        trigger_terms: List[str],
    ) -> bool:
        """Allow escalation policies only when the claim contains triggers."""

        if not trigger_terms:
            return False

        trigger_policy_prefixes = {
            "weekend": ["POL-WKND", "POL-MANUAL"],
            "policy exception": ["POL-MANUAL"],
            "luxury": ["POL-HOTEL", "POL-REJECT", "POL-MANUAL"],
            "suite": ["POL-HOTEL", "POL-REJECT", "POL-MANUAL"],
            "business class": ["POL-FLIGHT", "POL-MANUAL", "POL-REJECT"],
            "missing receipt": ["POL-RECEIPT", "POL-MANUAL"],
            "late submission": ["POL-TIME", "POL-REJECT", "POL-MANUAL"],
            "duplicate": ["POL-DUP", "POL-REJECT", "POL-MANUAL"],
        }

        allowed_prefixes = []
        for term in trigger_terms:
            allowed_prefixes.extend(trigger_policy_prefixes.get(term, []))

        if any(policy_id.startswith(prefix) for prefix in allowed_prefixes):
            return True

        return any(term in content for term in trigger_terms)

    def _manual_review_trigger_terms(self, claim: Dict) -> List[str]:
        """Extract only explicit escalation signals from the claim facts."""

        description = str(claim.get("description", "")).lower()
        terms = []

        if any(term in description for term in ["weekend", "saturday", "sunday"]):
            terms.append("weekend")

        if claim.get("is_policy_exception") or "policy exception" in description:
            terms.append("policy exception")

        if any(term in description for term in ["luxury", "ritz", "st. regis"]):
            terms.append("luxury")

        if "suite" in description:
            terms.append("suite")

        if "business class" in description:
            terms.append("business class")

        if not claim.get("receipt_uploaded"):
            terms.append("missing receipt")

        if "late submission" in description or "over 90 days" in description:
            terms.append("late submission")

        if "duplicate" in description:
            terms.append("duplicate")

        return terms

    @staticmethod
    def _primary_policy_hints(expense_type: str) -> str:
        """Return policy IDs that should be preferred for the category."""

        hints = {
            "hotel": "POL-HOTEL-001A, POL-HOTEL-002, POL-LIM-001",
            "meal": "POL-MEAL-001, POL-MEAL-002, POL-LIM-001",
            "laundry": "POL-LAUND-001, POL-LIM-001",
            "flight": "POL-FLIGHT-001, POL-FLIGHT-002",
            "transport": "POL-TRANS-001, POL-LIM-001",
        }

        return hints.get(str(expense_type).lower(), "POL-LIM-001")

    def _primary_policy_document(self, claim: Dict) -> Document | None:
        """Return a primary policy fallback when vector retrieval misses it."""

        expense_type = str(claim.get("expense_type", "")).lower()

        policies = {
            "hotel": (
                "POL-HOTEL-001A",
                "hotel",
                "Domestic Hotel Accommodation",
                (
                    "Employees traveling within India are eligible for "
                    "business-grade hotels. Tier 1 Indian cities including "
                    "Mumbai, Delhi, Bangalore, and Hyderabad have a maximum "
                    "hotel reimbursement limit of INR 8,000 per night "
                    "excluding taxes."
                ),
            ),
            "meal": (
                "POL-MEAL-001",
                "meal",
                "Meal Reimbursement Policy",
                (
                    "Domestic actual meal expense limits are INR 500 for "
                    "breakfast, INR 800 for lunch, and INR 1,200 for dinner. "
                    "International actual meal limits are USD 25, USD 40, "
                    "and USD 75 respectively."
                ),
            ),
            "laundry": (
                "POL-LAUND-001",
                "laundry",
                "Laundry Expenses",
                (
                    "Laundry expenses are reimbursable only for trips "
                    "exceeding 5 consecutive business days, up to USD 30 "
                    "per week."
                ),
            ),
        }

        policy = policies.get(expense_type)
        if not policy:
            return None

        policy_id, policy_type, section, content = policy
        return Document(
            page_content=content,
            metadata={
                "policy_id": policy_id,
                "policy_type": policy_type,
                "section": section,
                "chunk_id": f"fallback-{policy_id}",
            },
        )

    def _contains_primary_policy(
        self,
        documents: List[Any],
        claim: Dict,
    ) -> bool:
        """Return True when filtered documents already include category policy."""

        expense_type = str(claim.get("expense_type", "")).lower()
        primary_prefixes = {
            "hotel": ("POL-HOTEL",),
            "meal": ("POL-MEAL",),
            "laundry": ("POL-LAUND",),
        }.get(expense_type, ())

        return any(
            str(doc.metadata.get("policy_id", "")).upper().startswith(
                primary_prefixes
            )
            for doc in documents
        )

    def _contains_unmatched_escalation_policy(
        self,
        content: str,
        trigger_terms: List[str],
    ) -> bool:
        """Reject generic chunks containing unrelated escalation policies."""

        escalation_prefixes = [
            "POL-MANUAL",
            "POL-WKND",
            "POL-REJECT",
            "POL-DUP",
            "POL-TIME",
        ]

        has_escalation_policy = any(
            prefix in content.upper() for prefix in escalation_prefixes
        )

        if not has_escalation_policy:
            return False

        return not any(term in content for term in trigger_terms)
