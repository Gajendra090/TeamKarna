# KNOWLEDGE X — Project Architecture Documentation

Complete source-backed architecture documentation and interactive Archify visualizations for **KNOWLEDGE X**, generated without modifying any application source code.

## Interactive Visualization & Report Suite

Open `architecture-report.html` for the unified architecture report, or inspect each interactive Archify diagram directly in any browser:

| Artifact | Archify Type | Focus & Coverage |
| --- | --- | --- |
| [architecture-report.html](architecture-report.html) | Archify Explainer | Executive architecture overview, frontend/backend matrix, tech stack, APIs, security controls, code metrics & diagram index |
| [system-architecture.html](system-architecture.html) | System Architecture | High-level Streamlit monolith, 7 workspace pages, backend modules, SQLite store, user upload directories & Google Cloud APIs |
| [module-communication.html](module-communication.html) | Architecture | Complete dependency graph across `app.py`, all 7 `pages/*.py`, all 8 `modules/*.py`, `evaluation/`, SQLite & `uploads/` |
| [api-architecture.html](api-architecture.html) | Architecture | Internal Python contracts (no separate REST server), Gemini API + model fallback, Google OAuth 2.0/OIDC & SSRF URL firewall |
| [database-erd.html](database-erd.html) | Database ERD | Physical SQLite schema (`knowledge_x_history.db`: 5 tables, 4 indexes), application-enforced `user_id` isolation & `uploads/user_<hash>` |
| [data-flow.html](data-flow.html) | Data Flow (DFD) | 5-stage pipeline from Local PDF / Public URL / Camera Scan & queries through validation, AI compute, secret scrubbing & UI |
| [user-workflow.html](user-workflow.html) | User Workflow | End-to-end researcher journey across Auth, Ingestion, AI Exploration, Ambiguity Clarification & Guardrail Recovery |
| [seq-google-oauth.html](seq-google-oauth.html) | Sequence | Google OAuth 2.0 PKCE S256 login, token exchange, OIDC verification, signed session cookie persistence & logout revocation |
| [seq-ai-synthesis-pipeline.html](seq-ai-synthesis-pipeline.html) | Sequence | Multi-channel intake -> `analyze_documents()` with model fallback -> `detect_research_gaps()` -> `build_structural_graph_data()` -> SQLite |
| [seq-ambiguity-detector.html](seq-ambiguity-detector.html) | Sequence | Deterministic `<2`-document guard, 3-way ambiguity classification, clarification UI, query rewrite & cited QA |
| [code-analysis/analysis/repo.analysis.html](code-analysis/analysis/repo.analysis.html) | Code Analysis | Static Python complexity & dependency workbench across 18 Python source files (`0` parse errors, `0` cycles) |

## Architectural Highlights Verified Against Source Code

- **Runtime Topology**: Single-process **Streamlit 1.45+** monolithic Python web application (`app.py`) dispatching via `st.navigation(position="sidebar")` across 7 pages (`pages/*.py`) and importing 8 domain modules (`modules/*.py`) directly in-process.
- **No Separate REST API Server**: All internal communication occurs via typed Python dataclasses, Pydantic schemas, and domain exceptions (`AIServiceError`, `PDFProcessingError`, `ScanProcessingError`, `AuthenticationError`, `HistoryStorageError`). External API integrations consist of the **Google Gemini API** (`google-genai`), **Google OAuth 2.0 / OIDC** (`accounts.google.com`, `oauth2.googleapis.com`), and **SSRF-gated external PDF URLs**.
- **Persistence & User Isolation**: `knowledge_x_history.db` stores 5 SQLite tables (`research_sessions`, `ambiguity_history`, `auth_sessions`, `oauth_states`, `app_settings`) with 4 indexes. Authenticated Google users receive isolated upload directories (`uploads/user_<sha256(user_id)[:16]>/`) and `user_id`-filtered queries, while legacy/unauthenticated research defaults safely to `legacy_local` inside `uploads/`.
- **Tech Stack Packaging Note**: Dependencies declared in `requirement.txt` (`streamlit`, `google-genai`, `pymupdf`, `networkx`, `plotly`, `python-dotenv`, `pydantic`, `pandas`, `pillow`, `requests`, `numpy`) should also include `google-auth` and `itsdangerous` imported by `modules/auth.py` when packaging fresh environments.
