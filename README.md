# 🏗️ CivilVision AI – AI-Powered Construction Site Visual Inspection & Reporting Assistant

**CivilVision AI** is an intelligent visual inspection and safety reporting assistant tailored for civil engineering students, site supervisors, and inspection personnel. It leverages the official **Google GenAI Python SDK** and multimodal Gemini vision models to analyze visible conditions in construction site photographs, categorize observations into structured records, engage in conversational Q&A, and generate inspection reports.

---

## 🌟 Key Features Across Phases

### Phase 1: Core Multimodal Inspection & Confidence Framework
- **Primary & Fallback Engine**: `gemini-3.7-flash` (Primary) with automatic exponential backoff & jitter failover to `gemini-3.6-flash` (Fallback) on 503 high demand or socket aborts (`[WinError 10053]`).
- **UI-Only Fallback Isolation**: Model notices are strictly UI-rendered and never injected into prompts or conversational history.
- **Evidence-Based Confidence System**:
  - 🟢 **HIGH VISUAL CONFIDENCE**: Directly visible, unobstructed elements with adequate resolution/lighting.
  - 🟡 **MEDIUM VISUAL CONFIDENCE**: Distant or partially occluded observations with stated visual limitations.
  - 🔍 **REQUIRES PHYSICAL VERIFICATION**: Mandatory category for parameters 2D photos cannot determine (concrete strength, rebar diameter/cover, scaffold anchorage, load capacity, code compliance).
  - **No Numerical Scores**: Completely prohibits arbitrary percentage confidence scores (e.g., 95%).
- **Conversational Directness**: Answers simple follow-up questions directly and concisely without repeating 4-section report headers.

### Phase 2: Structured Construction Inspection
- **9 Controlled Inspection Categories**:
  1. `Structural / Concrete`
  2. `Formwork & Shoring`
  3. `Scaffolding`
  4. `Site Safety`
  5. `PPE`
  6. `Equipment / Machinery`
  7. `Materials`
  8. `Housekeeping / Site Conditions`
  9. `Work Progress`
- **Structured Observation Schema**: Every observation contains:
  - `category`
  - `observation`
  - `evidence_type` (`VISIBLE` | `INFERRED` | `NOT_DETERMINABLE`)
  - `visual_confidence` (`HIGH` | `MEDIUM` | `REQUIRES_PHYSICAL_VERIFICATION`)
  - `potential_issue`
  - `physical_verification_required`
  - `recommended_action`
  - `risk_priority` (`HIGH ATTENTION` | `MEDIUM ATTENTION` | `LOW ATTENTION`)
- **Interactive UI**:
  - Executive summary card
  - Attention level metric pills (🔴 High Attention, 🟡 Medium Attention, 🟢 Low Attention)
  - Priority & Category interactive filtering
  - High-contrast engineering observation cards

### Phase 3A: In-Memory Inspection History
- **InspectionRecord Model**: Rich structured dataclass encapsulating `inspection_id`, timestamp, site activity, summary, observations, and image metadata.
- **Unique Deterministic ID Generation**: Fast `CV-YYYYMMDD-XXXXXX` identifier format for every inspection.
- **Sidebar History Drawer**: Quick navigation through past inspection sessions within the active session.
- **Client-Side Filtering & Sorting**: Filter historical records by category, priority, and date without server re-computation.
- **Zero API Footprint**: In-memory history operations make zero Gemini/external network calls.

### Phase 3B: Inspection Comparison Engine
- **Deterministic Delta Analysis**: Compare two inspection records side-by-side to track changes across site visits.
- **Observation & Attention Metrics Diff**: Computes exact net changes in high, medium, and low attention counts.
- **Category-by-Category Shift Tracking**: Pinpoints changes in observation counts for each of the 9 engineering categories.
- **Neutral Change Summaries**: Factual, objective delta reporting without speculative claims.
- **Zero API Cost**: 100% computed locally in Python.

### Phase 3C: Local Inspection Report Foundation
- **Clean Report Representation**: Standardized `InspectionReportRepresentation` data model generated locally from stored inspection records.
- **Category-Grouped Observations**: Observations logically grouped and prioritized for formal site reporting.
- **Preserved Engineering Metadata**: Retains original IDs, timestamps, image attributes, and observation records without mutations.
- **Interactive In-App Preview**: Collapsible, professional preview directly in the application interface.
- **Deterministic & Offline**: Operates completely offline with zero API calls.

### Phase 3D: PDF Inspection Report Export
- **One-Click Native PDF Generation**: Deterministic export engine producing professional civil inspection documents.
- **Engineering-Grade Layout**: Clean typography, color-coded priority indicators, summary metrics table, and detailed observations.
- **Built-in Engineering Disclaimer**: Prominently highlights visual screening scope and mandatory physical verification rules.
- **Zero External Dependencies**: Pure Python byte-stream generation without requiring heavy external PDF rendering binaries.
- **Zero Gemini Calls**: Operates on existing structured data in memory.

### Phase 4A: Evidence & Inspection Attachments
- **Evidence-Type Breakdown**: Precise accounting of `VISIBLE`, `INFERRED`, and `NOT_DETERMINABLE` evidence across all observations.
- **Source Image Metadata Tracking**: Preserves filename, image dimensions (px), and file size (KB) without duplicating raw image bytes in history records.
- **Missing Historical Image Graceful Fallback**: Clearly communicates when an image was not retained while keeping the structured inspection record fully accessible.
- **Evidence Summary in PDF & Reports**: Embeds evidence-type distributions into exported reports and summaries.

### Phase 4B: Workspace & Workflow Hardening
- **Hardened Workspace Reset (`reset_current_workspace`)**: Clears transient upload and active analysis state while strictly preserving inspection history and application settings.
- **Historical Inspection Selection (`select_historical_inspection`)**: Allows viewing past inspection records without mutating session history or generating duplicates.
- **Zero Gemini Footprint**: All workspace resets and historical navigation occur completely offline without API usage.

### Reliability, API-Efficiency & Error Handling Hardening
- **Input Validation & Corrupt Image Handling**: Validates uploaded images with PIL integrity verification (`verify()`); detects empty/zero-byte files and unsupported formats gracefully before sending requests.
- **Stale State Prevention**: Replaces or uploads of new images invalidate transient structured observations, chat history, and active inspection IDs without triggering unintended Gemini analyses or mutating stored historical inspections.
- **Robust Model Output Parsing (`parse_inspection_json`)**: Gracefully handles non-dict JSON payloads, markdown code fencing (````json` and ````), empty inputs, missing summary fields, and non-conforming observation items without crashing or inventing engineering data.
- **Strict Evidence Boundaries**: Enforces controlled inspection categories, confidence levels, and risk priorities with safe engineering defaults; never invents dimensions, rebar diameters, concrete grades, or code compliance claims.
- **Zero Gemini Call Guarantee**: Deterministically enforces that history retrieval, history filtering, inspection comparison, local report representation, PDF generation, workspace reset, and historical selection make zero network or Gemini calls.
- **User-Facing Error Safety**: Hides raw tracebacks on API or PDF generation issues, delivering clear, actionable feedback to users.

### Persistent Local SQLite Inspection Storage
- **Built-in SQLite Persistence Layer (`storage.py`)**: Stores inspections locally in `civilvision_inspections.db` using Python's standard `sqlite3` module. No PostgreSQL, cloud database, or Docker required.
- **Normalized Relational Schema**: Persists inspections in `inspections` and individual observations in `observations` with a foreign-key relationship and cascade deletion.
- **Ordered Observation Fidelity**: Preserves original sequence ordering of observation records via `sequence_order` column.
- **Duplicate Protection**: Uses parameterized `ON CONFLICT(inspection_id)` upsert semantics to ensure the same inspection ID never creates duplicate rows in the database.
- **Session State vs. Database Distinction**: Active workspace state lives in Streamlit `session_state`, while historical inspections are preloaded from and persisted to SQLite on creation.
- **Workspace Reset Safety**: Workspace resets clear transient inputs and active analysis state while strictly preserving persisted database records.
- **Clean Local Developer Reset**: Developers can reset the local database by deleting the `civilvision_inspections.db` file; the schema will reinitialize automatically on the next startup.

### Structured CSV Inspection Data Export
- **One-Click Native CSV Export**: Directly export structured site observations (`export_inspection_to_csv`) to standard CSV format for spreadsheet analysis (Excel, Google Sheets), BIM integration, and project management databases.
- **11 Standardized Columns**: Exports `inspection_id`, `timestamp`, `site_activity`, `category`, `risk_priority`, `visual_confidence`, `evidence_type`, `observation`, `potential_issue`, `physical_verification_required`, and `recommended_action`.
- **RFC 4180 Escaping**: Safely handles commas, double quotes, and multi-line strings in observation text using Python's standard `csv` library.
- **Zero API Footprint**: Operates 100% locally from active and historical structured inspection records with zero Gemini or network calls.

---

## 🏛️ System Architecture

```
[Construction Site Image] (JPG, PNG, WEBP)
            │
            ▼
 ┌─────────────────────────────────────────┐
 │        Streamlit Web Application        │
 │  (Image preview, interactive chat, UI)  │
 └─────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │    Official Google GenAI Python SDK     │
 │  Primary:  gemini-3.7-flash             │
 │  Fallback: gemini-3.6-flash (on 503)    │
 └─────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │    Phase 2 Structured Schema Pipeline   │
 │   - Controlled 9-category inspection    │
 │   - Evidence boundary enforcement       │
 │   - Risk priority classification        │
 │   - Observation record cards & metrics  │
 └─────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │  Phase 3 History & Comparison Pipeline  │
 │   - In-memory inspection history (3A)   │
 │   - Side-by-side delta comparison (3B)  │
 │   - Local report representation (3C)    │
 │   - One-click PDF export engine (3D)    │
 └─────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │   Persistent SQLite Storage Layer       │
 │   - Local sqlite3 database (storage.py) │
 │   - Zero Gemini footprint guarantee     │
 │   - Duplicate ID protection             │
 │   - One-to-many normalized schema       │
 └─────────────────────────────────────────┘
            │
            ▼
  [Download: PDF Report / CSV Data / TXT Inspection Summary]
```

---

## 📁 Project Structure

```
CivilVision-AI/
├── app.py                     # Streamlit application UI, sidebar history & workflow orchestration
├── schema.py                  # Inspection data models, comparison, report representation, CSV & PDF export
├── storage.py                 # Local SQLite persistence engine, schema migrations & CRUD operations
├── prompts.py                 # System instructions, structured inspection & chat prompts
├── test_confidence_system.py  # 148 automated regression tests (100% deterministic & offline)
├── requirements.txt           # Application dependencies (streamlit, google-genai, pillow, reportlab)
├── README.md                  # Comprehensive technical documentation across all phases
├── .gitignore                 # Secrets, local database, and environment exclusions
└── .streamlit/
    ├── config.toml            # UI styling and dark theme configuration
    └── secrets.toml.example   # Template for Gemini API credentials
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Regression Test Suite
The complete regression test suite covers 148 tests across Phases 1, 2, 3A, 3B, 3C, 3D, 4A, 4B, Reliability Hardening, Local SQLite Persistence, and Structured CSV Export:
```bash
python test_confidence_system.py
```

### 4. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Civil Engineering & Safety Disclaimers

> **IMPORTANT NOTICE:**
> CivilVision AI is designed as an educational assistant and preliminary visual screening tool for construction sites.
> 1. **Visual Limitations**: 2D photographs cannot reveal internal concrete compaction, rebar yield strength, subsurface soil bearing capacity, or subterranean utilities.
> 2. **No Code Certification**: Visual analysis does NOT constitute a structural engineering sign-off, building code compliance certificate, or official safety permit.
> 3. **Mandatory Physical Verification**: Any observation or potential concern flagged by this tool must be independently verified on-site by a licensed Professional Engineer (PE) or certified safety inspector using calibrated physical testing methods (e.g., rebound hammer, cover meter, slump test, ultrasonic pulse velocity).
