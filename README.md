# 🏗️ CivilVision AI

## AI-Powered Construction Site Visual Inspection & Reporting Assistant

CivilVision AI is an AI-powered visual inspection and safety reporting assistant designed for civil engineering students, site supervisors, and inspection personnel.

It uses the official Google GenAI Python SDK and multimodal Gemini vision models to analyze construction-site photographs, identify visible conditions, organize observations into structured inspection records, support conversational follow-up questions, and generate inspection reports.

### 🚀 Live Demo

**[Open CivilVision AI →](https://civilvision-ai-uthwmqib2x7dqdus2czctr.streamlit.app/)**

### 🔐 Demo Login

Use these credentials to access the deployed demonstration application:

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `CivilVision2026!` |

---

## 🚀 Project Overview

CivilVision AI transforms construction-site images into structured visual inspection reports containing:

- Construction activity identification
- Engineering observation categorization
- Evidence classification
- Visual confidence assessment
- Risk-priority classification
- Physical verification requirements
- Recommended actions
- Inspection history
- Inspection comparison
- PDF report export
- CSV data export
- Historical source-image viewing
- Conversational follow-up analysis
- Authentication-protected application access

The system is designed as a **preliminary visual screening and decision-support tool**, not as a replacement for professional engineering inspection.

---

## ✨ Key Features

### 🤖 1. Multimodal AI Inspection

- Construction-site image analysis using Gemini vision models
- Primary and fallback model strategy
- Automatic retry handling for temporary model/API failures
- Evidence-based visual confidence classification
- Conversational follow-up questions
- Structured inspection generation

### 🏗️ 2. Structured Construction Inspection

CivilVision AI organizes observations into nine controlled categories:

1. Structural / Concrete
2. Formwork & Shoring
3. Scaffolding
4. Site Safety
5. PPE
6. Equipment / Machinery
7. Materials
8. Housekeeping / Site Conditions
9. Work Progress

Each observation contains:

- Category
- Observation
- Evidence type
- Visual confidence
- Potential issue
- Physical verification requirement
- Recommended action
- Risk priority

### 🔎 3. Evidence-Based Confidence

The system distinguishes between:

- 🟢 **HIGH** — Clearly visible evidence
- 🟡 **MEDIUM** — Partially visible, distant, or limited evidence
- 🔍 **REQUIRES PHYSICAL VERIFICATION** — Information that cannot reliably be determined from a 2D photograph

The system avoids inventing engineering measurements, reinforcement details, material properties, or code-compliance claims.

### 📚 4. Inspection History

- Unique inspection IDs
- Historical inspection navigation
- Persistent SQLite storage
- Category and priority filtering
- Historical source-image restoration when available
- Safe workspace reset
- Local history operations without additional Gemini API calls

### 🔄 5. Inspection Comparison

Two inspection records can be compared to identify:

- Changes in observation counts
- Changes in attention levels
- Category-level changes
- Deterministic inspection differences

Comparison is performed locally without additional Gemini API calls.

### 📄 6. PDF Inspection Reports

The application provides one-click PDF report generation containing:

- Inspection information
- Summary metrics
- Categorized observations
- Priority indicators
- Evidence information
- Engineering disclaimer

### 📊 7. CSV Inspection Data Export

Structured inspection observations can be exported to CSV for use with:

- Microsoft Excel
- Google Sheets
- BIM workflows
- Project-management databases

The export contains 11 standardized inspection fields and safely handles commas, quotation marks, and multiline text.

### 🖼️ 8. Historical Source Image Viewing

Historical inspections can restore cached source-image information when available, including:

- Original image bytes
- Image MIME type
- Original filename
- Historical inspection context

If an image is unavailable, the application gracefully falls back to the stored inspection information.

### 🔐 9. Authentication

The application includes a protected login system with:

- Streamlit Secrets-based credentials
- Session authentication
- Admin user indicator
- Logout functionality
- Protected application interface

---

## 🏛️ System Architecture

```text
Construction Site Image
          │
          ▼
┌──────────────────────────────┐
│     Streamlit Web App        │
│ Image Upload • Preview • UI  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Google GenAI SDK         │
│      Gemini Vision Models    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Structured Inspection Engine │
│ Categories • Evidence • Risk │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ SQLite Store │  │ Report Engine│
│ Inspection   │  │ PDF / CSV    │
│ History      │  │ Export       │
└──────────────┘  └──────────────┘
```

---

## 🧠 AI Inspection Workflow

```text
Upload Construction Image
          ↓
Validate Image
          ↓
Gemini Vision Analysis
          ↓
Structured Observation Extraction
          ↓
Evidence & Confidence Validation
          ↓
Risk Priority Classification
          ↓
Inspection Record Creation
          ↓
SQLite Persistence
          ↓
Interactive Dashboard
          ↓
PDF / CSV / Summary Export
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| AI / Vision | Google Gemini |
| SDK | Google GenAI Python SDK |
| Programming Language | Python |
| Database | SQLite |
| Image Processing | Pillow |
| PDF Generation | ReportLab |
| Data Export | Python CSV |
| Testing | Python Automated Test Suite |
| Deployment | Streamlit Community Cloud |
| Source Control | Git / GitHub |

---

## 📁 Project Structure

```text
CivilVision-AI/
│
├── app.py
├── schema.py
├── storage.py
├── prompts.py
├── test_confidence_system.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, authentication, image workflow, dashboard, history, and report controls |
| `schema.py` | Inspection data models, structured observations, comparison, PDF generation, and CSV export |
| `storage.py` | SQLite persistence and inspection database operations |
| `prompts.py` | AI inspection and conversational prompts |
| `test_confidence_system.py` | Automated regression and feature tests |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Local Development

### Prerequisites

- Python 3.10+
- Google Gemini API key
- Git

### Clone the Repository

```bash
git clone https://github.com/Goutham0104/CivilVision-AI.git
cd CivilVision-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Secrets

For local development, create:

```text
.streamlit/secrets.toml
```

Example:

```toml
GEMINI_API_KEY = "your_api_key"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "your_password"
```

**Never commit `secrets.toml` to GitHub.**

The repository contains `secrets.toml.example` as a configuration template.

---

## ▶️ Run Locally

Start the application with:

```bash
streamlit run app.py
```

Streamlit will provide a local development address when the application is running on your computer.

For the deployed application, use the **Live Demo** link at the top of this README.

---

## 🧪 Testing & Verification

The project includes automated tests covering:

- Core AI inspection functionality
- Structured observations
- Confidence and evidence handling
- Inspection history
- Inspection comparison
- PDF generation
- SQLite persistence
- CSV export
- Historical image viewing
- Authentication
- Reliability and workflow hardening

### Final Verification

The final verification confirmed:

- **158/158 automated tests passed**
- Python bytecode compilation completed successfully
- Streamlit `AppTest` health check completed successfully
- No runtime exceptions during the initial application render

Run the complete test suite with:

```bash
python test_confidence_system.py
```

Run the compilation check with:

```bash
python -m py_compile app.py schema.py storage.py prompts.py test_confidence_system.py
```

---

## 🔐 Security

Sensitive credentials are not stored in the public source code.

Deployment credentials should be configured through **Streamlit Secrets**.

The following file should never be committed:

```text
.streamlit/secrets.toml
```

API keys should never be published in:

- GitHub source code
- README files
- Screenshots
- Public documentation

> **Demo credential note:** The credentials listed above are intentionally provided for accessing the public demonstration application. Do not reuse them for any sensitive or production system.

---

## 📊 Engineering Safety & Limitations

CivilVision AI is an **educational assistant and preliminary visual screening tool**.

A photograph cannot reliably determine several engineering properties, including:

- Concrete compressive strength
- Reinforcement diameter
- Reinforcement cover
- Internal structural defects
- Subsurface conditions
- Load capacity
- Foundation performance
- Structural anchorage
- Complete code compliance

AI-generated observations must therefore be independently verified through appropriate physical inspection and engineering procedures.

CivilVision AI does **not** provide:

- Structural engineering certification
- Building-code certification
- Safety permits
- Professional engineering sign-off

---

## 🎯 Intended Users

CivilVision AI is intended to support:

- Civil engineering students
- Construction-site trainees
- Site supervisors
- Inspection personnel
- Project managers
- Engineering educators
- Construction technology researchers

---

## 🚧 Project Status

**Status: Deployed and Functional**

Current capabilities include:

- AI-powered visual inspection
- Structured construction observations
- Evidence and confidence classification
- Inspection history
- SQLite persistence
- Inspection comparison
- PDF report generation
- CSV data export
- Historical source-image viewing
- Authentication
- Streamlit Cloud deployment

---

## 👨‍💻 Repository & Demo

**GitHub Repository:**  
https://github.com/Goutham0104/CivilVision-AI

**Live Demo:**  
https://civilvision-ai-uthwmqib2x7dqdus2czctr.streamlit.app/

---

## ⚠️ Disclaimer

CivilVision AI provides AI-assisted visual screening based on available image evidence. It should not be treated as a substitute for qualified civil or structural engineering judgment, physical testing, site inspection, or applicable standards and regulations.

Any potentially unsafe condition identified by the system should be physically verified by an appropriately qualified professional.
