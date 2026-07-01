# 🤖 Enterprise AI Travel Reimbursement Agent

An enterprise-grade **Agentic AI Travel Reimbursement System** built using **LangGraph**, **Google Gemini**, **Retrieval-Augmented Generation (RAG)**, **ChromaDB**, and **Streamlit**.

The application automates employee travel reimbursement processing by retrieving company travel policies, invoking specialized business tools, and generating structured reimbursement decisions with audit trails and confidence scores.

---

# 📌 Problem Statement

Organizations spend significant time manually validating travel reimbursement claims against company policies. Manual processing is:

- Time consuming
- Error prone
- Difficult to audit
- Inconsistent across reviewers

This project automates the reimbursement workflow using an AI agent capable of reasoning over company policies and business rules while maintaining explainable decision making.

---

# 🚀 Features

- ✅ Streamlit-based Claim Submission UI
- ✅ LangGraph Multi-Agent Workflow
- ✅ Google Gemini Decision Making
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Chroma Vector Database
- ✅ Policy Retrieval from Enterprise Travel Policy
- ✅ Receipt Validation
- ✅ Expense Limit Checking
- ✅ Approval Threshold Validation
- ✅ Duplicate Claim Detection
- ✅ Confidence Score Calculation
- ✅ Manual Review Routing
- ✅ Structured JSON Output
- ✅ Audit Trail Generation
- ✅ Comprehensive Unit Tests

---

# 🏗️ System Architecture

```
                        Employee
                            │
                            ▼
                   Streamlit User Interface
                            │
                            ▼
                     Claim Submission
                            │
                            ▼
                  LangGraph AI Workflow
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
 Policy Retrieval       Google Gemini      Validation
      (RAG)               Decision             Node
        │
        ▼
  Chroma Vector Database
        │
        ▼
Travel Policy Knowledge Base
        │
        ▼
───────────────────────────────────────────────
Receipt Checker
Expense Limit Checker
Approval Checker
Duplicate Checker
Confidence Calculator
Explanation Generator
───────────────────────────────────────────────
        │
        ▼
 Structured Reimbursement Decision
        │
        ▼
     Streamlit Output
```

---

# 🧠 Agent Workflow

The LangGraph workflow follows these steps:

```
START
   │
   ▼
Validate Claim
   │
   ▼
Retrieve Company Policy (RAG)
   │
   ▼
Gemini Tool Selection
   │
   ├─────────────┐
   ▼             ▼
Receipt Checker  Expense Checker
        │
        ▼
Approval Checker
        │
        ▼
Duplicate Checker
        │
        ▼
Confidence Calculator
        │
        ▼
Final Decision
        │
        ▼
Explanation Generator
        │
        ▼
Structured JSON Output
        │
        ▼
END
```

---

# 📂 Project Structure

```
travel-reimbursement-agent/

│
├── agent/
│   ├── graph.py
│   ├── nodes.py
│   ├── prompts.py
│   └── state.py
│
├── rag/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── build_vector_db.py
│
├── tools/
│   ├── policy_retriever.py
│   ├── receipt_checker.py
│   ├── expense_limit_checker.py
│   ├── approval_checker.py
│   ├── duplicate_checker.py
│   ├── confidence_calculator.py
│   └── explanation_generator.py
│
├── schemas/
│
├── ui/
│
├── config/
│
├── data/
│
├── sample_data/
│
├── tests/
│
└── app.py
```

---

# ⚙️ Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| LLM | Google Gemini |
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| RAG | LangChain RAG |
| Vector Database | ChromaDB |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| UI | Streamlit |
| Validation | Pydantic |
| Testing | PyTest |

---

# 🔍 Retrieval-Augmented Generation (RAG)

The project uses Retrieval-Augmented Generation to ensure all reimbursement decisions are grounded in enterprise travel policies.

Pipeline:

```
Travel Policy
      │
      ▼
Document Loader
      │
      ▼
Markdown Text Splitter
      │
      ▼
Embedding Generation
      │
      ▼
Chroma Vector Database
      │
      ▼
Retriever
      │
      ▼
Gemini Context
```

---

# 🛠️ Business Tools

The AI agent dynamically invokes specialized tools during claim evaluation.

| Tool | Purpose |
|------|----------|
| Policy Retriever | Retrieves relevant policy context |
| Receipt Checker | Validates receipt availability |
| Expense Limit Checker | Checks reimbursement limits |
| Approval Checker | Validates approval requirements |
| Duplicate Checker | Detects duplicate submissions |
| Confidence Calculator | Calculates decision confidence |
| Explanation Generator | Generates explainable decisions |

---

# 📥 Supported Inputs

The application accepts reimbursement claims through the Streamlit interface.

Example fields include:

- Claim ID
- Employee ID
- Employee Role
- Expense Type
- Amount
- Currency
- Vendor
- Invoice Number
- Travel Type
- Approval Flags
- Receipt Status
- Business Description

---

# 📤 Sample Output

```json
{
  "claim_id": "CLM-1001",
  "decision": "Approve",
  "claim_status": "Approved",
  "approved_amount": 7500,
  "rejected_amount": 0,
  "currency": "INR",
  "policy_references": [
    "POL-HOTEL-001A"
  ],
  "confidence_score": "96%",
  "reviewer_required": false,
  "explanation": "Hotel expense complies with Tier-1 reimbursement policy.",
  "audit_trail": {
    "receipt_checked": true,
    "duplicate_checked": true,
    "expense_limit_checked": true,
    "approval_checked": true
  }
}
```

---

# 📋 Assessment Requirements Mapping

| Requirement | Status |
|------------|--------|
| Claim Intake | ✅ |
| Context Grounding (RAG) | ✅ |
| Tool Usage | ✅ |
| Agentic Workflow | ✅ |
| Structured Output | ✅ |
| Manual Review | ✅ |
| Streamlit Interface | ✅ |
| Audit Trail | ✅ |

---

# ▶️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Navigate to the project

```bash
cd travel-reimbursement-agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

Configure environment variables

```bash
cp .env.example .env
```

Add your Google Gemini API key inside `.env`.

Build the vector database

```bash
python -m rag.build_vector_db
```

Run the application

```bash
python app.py
```

or

```bash
streamlit run ui/streamlit_app.py
```

---

# 🧪 Running Tests

```bash
pytest
```

---

# 🔮 Future Enhancements

- OCR-based receipt verification
- PDF receipt upload and extraction
- Real ERP integration
- REST API endpoints
- Multi-policy support
- Human approval dashboard
- Analytics dashboard
- Multi-language support

---

# 👨‍💻 Author

Developed as part of an **Enterprise AI Travel Reimbursement Agent** assessment using modern Agentic AI, RAG, and LangGraph workflows.