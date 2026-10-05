"""
CivilVision AI – AI-Powered Construction Site Visual Inspection & Reporting Assistant
Streamlit Application
"""

import io
import os
import random
import time
from typing import Optional, Tuple
import streamlit as st
from PIL import Image

# Import prompt templates and system instructions
from prompts import (
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    MODEL,
    SYSTEM_INSTRUCTION,
    SITE_ANALYSIS_PROMPT,
    STRUCTURED_SITE_ANALYSIS_PROMPT,
    build_chat_prompt,
    SUMMARY_GENERATION_PROMPT,
)
from schema import (
    INSPECTION_CATEGORIES,
    VISUAL_CONFIDENCE_LEVELS,
    RISK_PRIORITIES,
    EVIDENCE_TYPES,
    ObservationRecord,
    InspectionReport,
    InspectionRecord,
    InspectionComparisonResult,
    InspectionReportRepresentation,
    generate_inspection_id,
    filter_inspection_records,
    compare_inspection_records,
    compare_category_counts,
    compare_priority_counts,
    compare_confidence_counts,
    generate_change_summary,
    generate_inspection_report,
    generate_inspection_pdf,
    parse_inspection_json,
    reset_current_workspace,
    select_historical_inspection,
    get_current_inspection,
    export_inspection_to_csv,
)
from storage import (
    initialize_database,
    save_inspection_record,
    load_inspection_record,
    load_inspection_history,
    inspection_exists,
    get_database_status,
)


# ---------------------------------------------------------
# Page Configuration & UI Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="CivilVision AI – Construction Site Visual Inspection Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, badges, and high-readability civil engineering styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #0F172A;
    }
    
    /* Hero Header Banner */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        color: #F59E0B;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    
    .hero-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Confidence Badges */
    .badge-obs {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    .badge-concern {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    
    .badge-verify {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* Phase 2 Risk Priority Badges */
    .prio-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.3px;
    }
    .prio-high {
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid #EF4444;
    }
    .prio-med {
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    .prio-low {
        background-color: rgba(34, 197, 94, 0.18);
        color: #4ADE80;
        border: 1px solid #22C55E;
    }
    
    /* Engineering Card */
    .eng-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .summary-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #CBD5E1;
        font-size: 0.85rem;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    
    /* Quick prompt button container */
    .quick-prompts {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    /* Login Page Styling */
    .login-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin: 20px auto;
        max-width: 520px;
    }
    .login-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .login-title {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .login-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Authentication Helpers & State
# ---------------------------------------------------------
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "CivilVision2026!"


def get_auth_credentials() -> Tuple[str, str]:
    """Retrieve demo/admin credentials from Streamlit secrets, with secure fallback."""
    user = DEFAULT_ADMIN_USER
    passwd = DEFAULT_ADMIN_PASS
    try:
        if "ADMIN_USERNAME" in st.secrets and st.secrets["ADMIN_USERNAME"]:
            user = str(st.secrets["ADMIN_USERNAME"]).strip()
        if "ADMIN_PASSWORD" in st.secrets and st.secrets["ADMIN_PASSWORD"]:
            passwd = str(st.secrets["ADMIN_PASSWORD"]).strip()
    except Exception:
        pass
    return user, passwd


def verify_login_credentials(input_user: str, input_pass: str) -> bool:
    """Verify input credentials against configured secrets."""
    expected_user, expected_pass = get_auth_credentials()
    return (
        bool(input_user)
        and bool(input_pass)
        and input_user.strip() == expected_user
        and input_pass.strip() == expected_pass
    )


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "image_mime" not in st.session_state:
    st.session_state.image_mime = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "structured_inspection" not in st.session_state:
    st.session_state.structured_inspection = None
if "analysis_fallback" not in st.session_state:
    st.session_state.analysis_fallback = None
if "summary_fallback" not in st.session_state:
    st.session_state.summary_fallback = None
if "last_chat_fallback" not in st.session_state:
    st.session_state.last_chat_fallback = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "inspection_summary" not in st.session_state:
    st.session_state.inspection_summary = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "inspection_history" not in st.session_state:
    # Initialize local SQLite storage and preload persisted inspections into active session
    try:
        initialize_database()
        st.session_state.inspection_history = load_inspection_history()
    except Exception:
        st.session_state.inspection_history = []
if "current_inspection_id" not in st.session_state:
    st.session_state.current_inspection_id = None
if "selected_historical_inspection_id" not in st.session_state:
    st.session_state.selected_historical_inspection_id = None
if "historical_image_cache" not in st.session_state:
    st.session_state.historical_image_cache = {}
if "active_inspection_image" not in st.session_state:
    st.session_state.active_inspection_image = None


# ---------------------------------------------------------
# Model & Fallback Configuration
# ---------------------------------------------------------
AVAILABLE_MODELS = [
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
]
FALLBACK_NOTICE_TEXT = (
    "Gemini 3.7 Flash is temporarily busy. CivilVision switched to Gemini 3.6 Flash for this request."
)


def is_transient_retryable_error(e: Exception) -> bool:
    """
    Identifies transient errors that can be retried:
    - 503 Service Unavailable / High Demand
    - Network / Socket resets (WinError 10053, ConnectionResetError, RemoteDisconnected, httpx.NetworkError)
    - Temporary server timeouts / 502 / 504
    Explicitly excludes permanent client errors (400, 401, 403, 404, invalid keys).
    """
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code == 503:
        return True
    if code in (400, 401, 403, 404):
        return False

    msg = str(e).lower()
    non_transient = [
        "api key not valid",
        "api_key_invalid",
        "permission_denied",
        "permission denied",
        "invalid_argument",
        "unauthenticated",
        "forbidden",
        "not found",
    ]
    if any(k in msg for k in non_transient):
        return False

    transient = [
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "server error",
        "resource has been exhausted",
        "resource_exhausted",
        "10053",
        "wsaeconnaborted",
        "connection reset",
        "connection aborted",
        "remoteprotocolerror",
        "networkerror",
        "connecterror",
        "timeout",
        "eof occurred in violation of protocol",
    ]
    return any(k in msg for k in transient)


def _call_model_with_retry(
    client,
    model_name: str,
    contents: list,
    config,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    base_delay: float = 1.0,
) -> Tuple[Optional[str], Optional[Exception], any]:
    """
    Executes a generate_content call for a given model with exponential backoff, jitter,
    and automatic fresh client instantiation on Windows socket/connection drops (WinError 10053).
    """
    from google import genai

    active_client = client
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            if active_client is None and api_key:
                active_client = genai.Client(api_key=api_key)

            response = active_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response.text, None, active_client
        except Exception as e:
            last_err = e
            if not is_transient_retryable_error(e):
                return None, e, active_client

            # On socket abort or connection drop, discard stale socket pool by creating fresh client
            if api_key and (
                "10053" in str(e)
                or "connection" in str(e).lower()
                or "remoteprotocolerror" in str(e).lower()
                or "protocol" in str(e).lower()
                or "network" in str(e).lower()
            ):
                try:
                    active_client = genai.Client(api_key=api_key)
                except Exception:
                    pass

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 0.4)
                time.sleep(delay)
            else:
                break

    return None, last_err, active_client


def generate_content_with_retry_and_fallback(
    client,
    model: str,
    contents: list,
    config,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    base_delay: float = 1.0,
) -> Tuple[str, bool, Optional[str]]:
    """
    Calls client.models.generate_content with:
    1. Exponential backoff and jitter on 503 / Network errors for primary model.
    2. Automatic fallback to FALLBACK_MODEL (gemini-3.6-flash) if PRIMARY_MODEL (gemini-3.7-flash) fails with 503/transient error.
    3. Exponential backoff and jitter on fallback model (bounded retry count).
    4. Client refresh on socket/connection aborts (WinError 10053).
    5. UI-only fallback notice return (never injected into prompt or output text).
    6. Non-transient errors raised immediately.

    Returns:
        (response_text, used_fallback, fallback_notice)
    """
    from google import genai

    active_client = client
    target_model = model

    # Step 1: Attempt target model (e.g. PRIMARY_MODEL) with exponential backoff
    resp_text, primary_err, active_client = _call_model_with_retry(
        client=active_client,
        model_name=target_model,
        contents=contents,
        config=config,
        api_key=api_key,
        max_retries=max_retries,
        base_delay=base_delay,
    )

    if resp_text is not None:
        return resp_text, False, None

    # If primary failed with non-transient error (e.g. invalid API key), raise immediately
    if not is_transient_retryable_error(primary_err):
        raise primary_err

    # Step 2: Fallback to FALLBACK_MODEL if target_model was PRIMARY_MODEL
    if target_model == PRIMARY_MODEL:
        # Re-initialize clean client instance for fallback if api_key is available
        if api_key:
            try:
                active_client = genai.Client(api_key=api_key)
            except Exception:
                pass

        fb_text, fb_err, active_client = _call_model_with_retry(
            client=active_client,
            model_name=FALLBACK_MODEL,
            contents=contents,
            config=config,
            api_key=api_key,
            max_retries=max_retries,
            base_delay=base_delay,
        )

        if fb_text is not None:
            return fb_text, True, FALLBACK_NOTICE_TEXT

        raise Exception(
            f"Both Gemini 3.7 Flash and Gemini 3.6 Flash are temporarily unavailable due to high demand. Please try again shortly."
        )

    raise primary_err


# ---------------------------------------------------------
# Google GenAI Client Helper (Cached)
# ---------------------------------------------------------
def get_api_key() -> str:
    """Retrieve Gemini API key from secrets, environment, or sidebar session state."""
    # 1. Check Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    # 2. Check environment variable
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # 3. Check session state (manual entry in sidebar)
    return st.session_state.get("user_api_key", "").strip()


@st.cache_resource(show_spinner=False)
def init_genai_client(api_key: str):
    """
    Initializes and caches the official Google GenAI Client.
    Using current official SDK: google-genai
    """
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Google GenAI Client: {e}")
        return None


# ---------------------------------------------------------
# Authentication Gate / Login Page
# ---------------------------------------------------------
def render_login_page():
    """Renders a simple, elegant CivilVision AI login page."""
    # Center column layout
    _, col_login, _ = st.columns([1, 1.4, 1])
    with col_login:
        st.markdown(
            """
            <div class="login-box">
                <div class="login-header">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">🏗️</div>
                    <h2 class="login-title">CivilVision AI</h2>
                    <div class="login-subtitle">Construction Site Visual Inspection & Reporting Assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("civilvision_login_form"):
            st.markdown("##### 🔐 Sign in to Workspace")
            input_user = st.text_input(
                "Username",
                placeholder="admin",
                key="login_username_input",
                help="Configured in Streamlit secrets (.streamlit/secrets.toml)",
            )
            input_pass = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••••••",
                key="login_password_input",
            )
            submit_login = st.form_submit_button(
                "Sign In 🚀",
                type="primary",
                use_container_width=True,
            )

            if submit_login:
                if verify_login_credentials(input_user, input_pass):
                    st.session_state.authenticated = True
                    st.session_state.auth_user = input_user.strip()
                    st.success("✅ Authentication successful. Loading workspace...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check your username and password.")

        # Convenient demo credential hint
        configured_user, _ = get_auth_credentials()
        st.markdown(
            f"""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:12px; margin-top:16px; font-size:0.8rem; color:#94A3B8; text-align:center;">
                <b>Demo Access:</b> Username: <code style="color:#F59E0B;">{configured_user}</code> &nbsp;|&nbsp; Credentials configured in Streamlit secrets.
            </div>
            """,
            unsafe_allow_html=True,
        )


# If user is not yet authenticated, render login page and stop execution of main app
if not st.session_state.get("authenticated", False):
    render_login_page()
    st.stop()


# ---------------------------------------------------------
# Sidebar Configuration & Tools
# ---------------------------------------------------------
with st.sidebar:
    # Authenticated user indicator and logout
    user_name = st.session_state.get("auth_user") or "admin"
    col_user, col_logout = st.columns([2.5, 1.5])
    with col_user:
        st.markdown(f"👤 **{user_name}** *(Admin)*")
    with col_logout:
        if st.button("Logout", key="btn_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_user = None
            st.rerun()
    st.markdown("---")

    st.markdown("### ⚙️ Engine Settings")

    api_key_configured = bool(get_api_key())
    
    # Model Selection
    selected_model = st.selectbox(
        "Gemini Multimodal Model",
        options=AVAILABLE_MODELS,
        index=0,
        help="Primary model: Gemini 3.7 Flash with automatic fallback to Gemini 3.6 Flash on high demand.",
    )

    # API Key Input if not set in secrets/env
    if not api_key_configured:
        st.warning("⚠️ API Key not detected in secrets or environment.")
        user_key = st.text_input(
            "Enter Google Gemini API Key:",
            type="password",
            placeholder="AIzaSy...",
            key="user_api_key_input",
            help="Get your key from Google AI Studio: https://aistudio.google.com/",
        )
        if user_key:
            st.session_state.user_api_key = user_key
            st.rerun()
    else:
        st.success("🔒 API Key Configured & Ready")

    st.markdown("---")
    st.markdown("### 📐 Evidence-Based Confidence System")
    st.markdown(
        """
        <div style="font-size: 0.85rem; line-height: 1.6;">
        <span class="badge-obs">🟢 HIGH VISUAL CONFIDENCE</span><br>
        Direct visual evidence, unobstructed view, adequate lighting.<br><br>
        <span class="badge-concern">🟡 MEDIUM VISUAL CONFIDENCE</span><br>
        Potential condition with stated visual ambiguity or distance.<br><br>
        <span class="badge-verify">🔍 REQUIRES PHYSICAL VERIFICATION</span><br>
        Engineering parameters that 2D photographs cannot determine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚠️ Engineering Disclaimers")
    st.markdown(
        """
        <div class="disclaimer-box">
        <b>Strict Evidence-Based Rule:</b><br>
        CivilVision AI strictly separates visual observations, visual confidence, and mandatory physical engineering verification. It never fabricates dimensions, concrete grades, or structural capacities.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Phase 3A: Inspection History (Sidebar Review & Navigation)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🗂️ Inspection History")
    history_records = st.session_state.get("inspection_history", [])
    if not history_records:
        st.caption("No previous inspections in current session.")
    else:
        st.caption(f"{len(history_records)} inspection(s) recorded in this session.")
        
        # Local Filtering Controls for History
        hist_sort = st.radio(
            "Sort Order",
            options=["Latest First", "Oldest First"],
            index=0,
            horizontal=True,
            key="hist_sort_radio",
        )
        hist_cat_filter = st.selectbox(
            "History Category Filter",
            options=["All Categories"] + sorted(list({o.category for r in history_records for o in r.observations})),
            key="hist_cat_select",
        )
        hist_prio_filter = st.selectbox(
            "History Priority Filter",
            options=["All Priorities", "HIGH ATTENTION", "MEDIUM ATTENTION", "LOW ATTENTION"],
            key="hist_prio_select",
        )

        filtered_history = filter_inspection_records(
            records=history_records,
            category=hist_cat_filter,
            priority=hist_prio_filter,
            reverse_chronological=(hist_sort == "Latest First"),
        )

        if not filtered_history:
            st.info("No inspections match the history filter criteria.")
        else:
            with st.container(height=340):
                for rec in filtered_history:
                    is_current = (rec.inspection_id == st.session_state.get("current_inspection_id"))
                    border_color = "#3B82F6" if is_current else "#334155"
                    bg_color = "rgba(59, 130, 246, 0.08)" if is_current else "#1E293B"
                    
                    st.markdown(
                        f"""
                        <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:10px; margin-bottom:8px;">
                            <div style="font-weight:700; font-size:0.85rem; color:#F59E0B; word-break:break-all;">
                                📋 {rec.inspection_id} {'<span style="color:#60A5FA;">(Active)</span>' if is_current else ''}
                            </div>
                            <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">
                                🕒 {rec.timestamp}
                            </div>
                            <div style="font-size:0.78rem; color:#E2E8F0; margin-top:4px;">
                                🏗️ {rec.site_activity[:60] + ('...' if len(rec.site_activity) > 60 else '')}
                            </div>
                            <div style="font-size:0.75rem; color:#CBD5E1; margin-top:6px; display:flex; gap:8px;">
                                <span>Total: <b>{rec.total_observations}</b></span>
                                <span style="color:#EF4444;">🔴 {rec.high_attention_count}</span>
                                <span style="color:#F59E0B;">🟡 {rec.medium_attention_count}</span>
                                <span style="color:#10B981;">🟢 {rec.low_attention_count}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Open Inspection button
                    if st.button(f"🔍 Open {rec.inspection_id[:16]}...", key=f"btn_open_{rec.inspection_id}", use_container_width=True):
                        # Preserve currently active image before switching away if not already viewing history
                        if not st.session_state.get("selected_historical_inspection_id") and st.session_state.get("image_bytes"):
                            st.session_state.active_inspection_image = {
                                "bytes": st.session_state.image_bytes,
                                "mime": st.session_state.image_mime,
                                "name": st.session_state.image_name,
                            }
                        selected = select_historical_inspection(st.session_state, rec.inspection_id)
                        target_rec = selected or rec
                        st.session_state.structured_inspection = target_rec.to_report()
                        st.session_state.current_inspection_id = target_rec.inspection_id

                        # Restore cached source image for this historical inspection if available
                        img_cache = st.session_state.get("historical_image_cache", {})
                        if rec.inspection_id in img_cache:
                            cached_img = img_cache[rec.inspection_id]
                            st.session_state.image_bytes = cached_img.get("bytes")
                            st.session_state.image_mime = cached_img.get("mime")
                            st.session_state.image_name = cached_img.get("name")
                        else:
                            st.session_state.image_bytes = None
                            st.session_state.image_mime = None
                            st.session_state.image_name = target_rec.source_image_name

                        # Format text context for chat
                        if target_rec.observations:
                            md_lines = [
                                f"### 🏗️ Construction Activity: {target_rec.site_activity}\n",
                                f"**Executive Summary:** {target_rec.executive_summary}\n",
                                "### 📋 Structured Inspection Observations:"
                            ]
                            for idx, obs in enumerate(target_rec.observations, 1):
                                md_lines.append(
                                    f"\n**{idx}. [{obs.category}] ({obs.risk_priority} | {obs.visual_confidence})**\n"
                                    f"- Observation ({obs.evidence_type}): {obs.observation}\n"
                                    f"- Potential Issue / Risk: {obs.potential_issue}\n"
                                    f"- Physical Verification Required: {obs.physical_verification_required}\n"
                                    f"- Recommended Action: {obs.recommended_action}"
                                )
                            st.session_state.analysis_result = "\n".join(md_lines)
                        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset / Clear Session", use_container_width=True):
        reset_current_workspace(st.session_state)
        st.rerun()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">🏗️ CivilVision AI</h1>
        <div class="hero-subtitle">AI-Powered Construction Site Visual Inspection & Reporting Assistant</div>
        <p class="hero-desc">
            Upload a construction-site image, analyse visible conditions using AI, ask questions, and generate an inspection summary.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Navigation Tabs: Active Inspection & Inspection Comparison
# ---------------------------------------------------------
tab_inspect, tab_compare = st.tabs(["🏗️ Current Site Inspection", "📊 Inspection Comparison"])

with tab_inspect:
    # Main Two-Column Layout (Upload & Inspection / Chat & Report)
    col_left, col_right = st.columns([1, 1], gap="large")

# ---------------------------------------------------------
# LEFT COLUMN: Image Upload & Visual Analysis
# ---------------------------------------------------------
with col_left:
    st.markdown("### 📷 Construction Site Image Upload")

    uploaded_file = st.file_uploader(
        "Choose a construction site photograph",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photograph of rebar, formwork, scaffolding, excavation, or ongoing site activity.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        # Validate non-empty file
        if not file_bytes or len(file_bytes) == 0:
            st.error("⚠️ The uploaded file is empty. Please upload a valid image file.")
            st.session_state.image_bytes = None
            st.session_state.image_mime = None
            st.session_state.image_name = None
        else:
            # Check for image replacement to clear stale transient analysis state
            if st.session_state.get("image_name") != uploaded_file.name:
                st.session_state.structured_inspection = None
                st.session_state.analysis_result = None
                st.session_state.analysis_fallback = None
                st.session_state.summary_fallback = None
                st.session_state.last_chat_fallback = None
                st.session_state.chat_history = []
                st.session_state.inspection_summary = None
                st.session_state.current_inspection_id = None

            # Verify image format and integrity with PIL
            valid_image = False
            pil_img = None
            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                pil_img.verify()  # Verify integrity
                # Reopen after verify because verify() exhausts image buffer
                pil_img = Image.open(io.BytesIO(file_bytes))
                valid_image = True
            except Exception:
                st.error("⚠️ The uploaded file could not be recognized as a valid image. Supported formats: JPG, JPEG, PNG, WEBP.")
                st.session_state.image_bytes = None
                st.session_state.image_mime = None
                st.session_state.image_name = None

            if valid_image and pil_img is not None:
                st.session_state.image_bytes = file_bytes
                st.session_state.image_mime = uploaded_file.type or "image/jpeg"
                st.session_state.image_name = uploaded_file.name

                st.image(
                    pil_img,
                    caption=f"Uploaded: {uploaded_file.name} ({pil_img.width}x{pil_img.height} px, {len(file_bytes)/1024:.1f} KB)",
                    use_container_width=True,
                )

                # Analysis Trigger Button
                analyse_button = st.button(
                    "🔍 Analyse Site Image",
                    type="primary",
                    use_container_width=True,
                )

                if analyse_button:
                    api_key = get_api_key()
                    if not api_key:
                        st.error("❌ Google Gemini API Key required. Please provide it in the sidebar or `.streamlit/secrets.toml`.")
                    else:
                        with st.spinner("Analyzing site image with Gemini Vision (Phase 2 Structured Inspection)..."):
                            try:
                                from google import genai
                                from google.genai import types

                                client = init_genai_client(api_key)
                                if client:
                                    image_part = types.Part.from_bytes(
                                        data=st.session_state.image_bytes,
                                        mime_type=st.session_state.image_mime,
                                    )
                                    
                                    response_text, used_fallback, fallback_notice = generate_content_with_retry_and_fallback(
                                        client=client,
                                        model=selected_model,
                                        contents=[image_part, STRUCTURED_SITE_ANALYSIS_PROMPT],
                                        config=types.GenerateContentConfig(
                                            system_instruction=SYSTEM_INSTRUCTION,
                                            temperature=0.2,
                                        ),
                                        api_key=api_key,
                                    )
                                    
                                    report = parse_inspection_json(response_text)
                                    st.session_state.structured_inspection = report
                                    st.session_state.analysis_fallback = fallback_notice if used_fallback else None

                                    # Phase 3A & Local Persistence: Store in session history and SQLite
                                    if report:
                                        img_dims = f"{pil_img.width}x{pil_img.height} px"
                                        img_size_kb = round(len(st.session_state.image_bytes) / 1024, 1)
                                        insp_record = InspectionRecord.from_report(
                                            report=report,
                                            source_image_name=st.session_state.image_name,
                                            source_image_size_kb=img_size_kb,
                                            source_image_dimensions=img_dims,
                                        )
                                        st.session_state.current_inspection_id = insp_record.inspection_id
                                        # Avoid duplicate insertion into session history
                                        if not any(r.inspection_id == insp_record.inspection_id for r in st.session_state.inspection_history):
                                            st.session_state.inspection_history.append(insp_record)

                                        # Persist to local SQLite storage
                                        try:
                                            save_inspection_record(insp_record)
                                        except Exception:
                                            pass

                                        # Cache source image for historical inspection review
                                        if "historical_image_cache" not in st.session_state:
                                            st.session_state.historical_image_cache = {}
                                        st.session_state.historical_image_cache[insp_record.inspection_id] = {
                                            "bytes": st.session_state.image_bytes,
                                            "mime": st.session_state.image_mime,
                                            "name": st.session_state.image_name,
                                        }
                                        st.session_state.active_inspection_image = {
                                            "bytes": st.session_state.image_bytes,
                                            "mime": st.session_state.image_mime,
                                            "name": st.session_state.image_name,
                                        }


                                    # Format textual representation for chat context and report synthesis
                                    if report and report.observations:
                                        md_lines = [
                                            f"### 🏗️ Construction Activity: {report.site_activity or report.summary}\n",
                                            f"**Executive Summary:** {report.summary}\n",
                                            "### 📋 Structured Inspection Observations:"
                                        ]
                                        for idx, obs in enumerate(report.observations, 1):
                                            md_lines.append(
                                                f"\n**{idx}. [{obs.category}] ({obs.risk_priority} | {obs.visual_confidence})**\n"
                                                f"- Observation ({obs.evidence_type}): {obs.observation}\n"
                                                f"- Potential Issue / Risk: {obs.potential_issue}\n"
                                                f"- Physical Verification Required: {obs.physical_verification_required}\n"
                                                f"- Recommended Action: {obs.recommended_action}"
                                            )
                                        st.session_state.analysis_result = "\n".join(md_lines)
                                    else:
                                        st.session_state.analysis_result = response_text or "No observations generated."

                                    # Reset chat history with clean greeting
                                    st.session_state.chat_history = [
                                        {
                                            "role": "assistant",
                                            "content": "✅ **Structured Visual Inspection Completed.**\n\nI have evaluated visible site conditions using the CivilVision AI Phase 2 Structured Inspection Framework. You can review the structured observation cards on the left, ask follow-up questions in the chat, or generate an inspection summary report.",
                                        }
                                    ]
                                    st.success("✅ Structured visual inspection complete!")
                            except Exception as e:
                                err_msg = str(e)
                                if "10053" in err_msg or "connection" in err_msg.lower():
                                    st.error("⚠️ Connection interrupted by local host network ([WinError 10053]). Please retry the analysis.")
                                elif "temporarily unavailable" in err_msg.lower() or "high demand" in err_msg.lower():
                                    st.error("⚠️ Gemini services are currently experiencing high demand. Please wait a few moments and retry.")
                                elif "api key" in err_msg.lower() or "unauthenticated" in err_msg.lower():
                                    st.error("❌ Invalid or missing Google Gemini API key. Please check your key in the sidebar.")
                                else:
                                    st.error(f"⚠️ Analysis could not be completed: {type(e).__name__} occurred.")


    else:
        st.info("👆 Please upload a construction site photograph (JPG, JPEG, PNG, or WEBP) to begin visual inspection.")

    # ---------------------------------------------------------
    # Phase 4A: Evidence / Source Image Section
    # ---------------------------------------------------------
    curr_id = st.session_state.get("current_inspection_id")
    active_hist_rec = None
    if curr_id:
        active_hist_rec = next(
            (r for r in st.session_state.get("inspection_history", []) if r.inspection_id == curr_id),
            None,
        )

    if curr_id or st.session_state.get("structured_inspection"):
        st.markdown("---")
        st.markdown("### 🖼️ Evidence / Source Image")
        if st.session_state.get("image_bytes"):
            try:
                pil_img = Image.open(io.BytesIO(st.session_state.image_bytes))
                dim_str = f"{pil_img.width}x{pil_img.height} px"
                size_kb_str = f"{len(st.session_state.image_bytes) / 1024:.1f} KB"
                fn_str = st.session_state.get("image_name") or (active_hist_rec.source_image_name if active_hist_rec else "source_image.jpg")
                st.image(
                    pil_img,
                    caption=f"Evidence Image: {fn_str} ({dim_str}, {size_kb_str})",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error displaying evidence image preview: {e}")
        else:
            # Historical inspection opened without raw image bytes in current session
            st.info("Source image preview is unavailable for this historical inspection. The structured inspection record remains available.")
            if active_hist_rec and (active_hist_rec.source_image_name or active_hist_rec.source_image_dimensions):
                meta_parts = []
                if active_hist_rec.source_image_name:
                    meta_parts.append(f"<b>Filename:</b> {active_hist_rec.source_image_name}")
                if active_hist_rec.source_image_dimensions:
                    meta_parts.append(f"<b>Dimensions:</b> {active_hist_rec.source_image_dimensions}")
                if active_hist_rec.source_image_size_kb:
                    meta_parts.append(f"<b>Size:</b> {active_hist_rec.source_image_size_kb} KB")
                st.markdown(
                    f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-size:0.8rem; color:#94A3B8; margin-top:6px;">
                        {' &nbsp;|&nbsp; '.join(meta_parts)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Display Structured Inspection Results if available
    if st.session_state.get("structured_inspection") and st.session_state.structured_inspection.observations:
        report = st.session_state.structured_inspection
        st.markdown("---")
        curr_id = st.session_state.get("current_inspection_id")
        sel_hist_id = st.session_state.get("selected_historical_inspection_id")
        is_viewing_history = bool(sel_hist_id and sel_hist_id == curr_id)

        if is_viewing_history:
            mode_badge = '<span style="font-size:0.78rem; background:rgba(245, 158, 11, 0.18); border:1px solid #F59E0B; padding:2px 8px; border-radius:4px; color:#FBBF24; margin-left:8px;">📖 Historical Inspection (Read-Only)</span>'
        else:
            mode_badge = '<span style="font-size:0.78rem; background:rgba(34, 197, 94, 0.18); border:1px solid #22C55E; padding:2px 8px; border-radius:4px; color:#4ADE80; margin-left:8px;">🟢 Active Inspection</span>'

        id_badge = f'<span style="font-size:0.8rem; background:#1E293B; border:1px solid #475569; padding:2px 8px; border-radius:4px; color:#F59E0B; margin-left:8px;">ID: {curr_id}</span>' if curr_id else ''
        st.markdown(f"### 📋 Structured Construction Visual Inspection {id_badge} {mode_badge}", unsafe_allow_html=True)

        if is_viewing_history:
            col_notice, col_return = st.columns([3, 1])
            with col_notice:
                st.info(f"Viewing historical inspection record `{curr_id}`. This record is immutable and read-only.")
            with col_return:
                if st.button("⬅️ Return to Active", use_container_width=True, key="btn_return_active"):
                    st.session_state.selected_historical_inspection_id = None
                    active_rec = get_current_inspection(st.session_state)
                    if active_rec:
                        st.session_state.structured_inspection = active_rec.to_report()
                        st.session_state.current_inspection_id = active_rec.inspection_id
                    # Restore active image if preserved
                    if st.session_state.get("active_inspection_image"):
                        act_img = st.session_state["active_inspection_image"]
                        st.session_state.image_bytes = act_img.get("bytes")
                        st.session_state.image_mime = act_img.get("mime")
                        st.session_state.image_name = act_img.get("name")
                    st.rerun()

        if st.session_state.get("analysis_fallback"):
            st.warning(f"⚠️ {st.session_state.analysis_fallback}")


        # Executive Summary Card
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="font-size: 0.95rem; color: #F59E0B; font-weight: 700; margin-bottom: 4px;">
                    🏗️ Activity Observed: {report.site_activity or 'Active Construction Work'}
                </div>
                <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
                    <b>Executive Summary:</b> {report.summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric Counters (Attention Priority)
        obs_list = report.observations
        high_cnt = sum(1 for o in obs_list if o.risk_priority == "HIGH ATTENTION")
        med_cnt = sum(1 for o in obs_list if o.risk_priority == "MEDIUM ATTENTION")
        low_cnt = sum(1 for o in obs_list if o.risk_priority == "LOW ATTENTION")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Items", len(obs_list))
        m_col2.metric("🔴 High Attn", high_cnt)
        m_col3.metric("🟡 Med Attn", med_cnt)
        m_col4.metric("🟢 Low Attn", low_cnt)

        # Phase 4A Evidence Summary Metrics (Calculated 100% locally, Zero Gemini calls)
        vis_cnt = sum(1 for o in obs_list if o.evidence_type == "VISIBLE")
        inf_cnt = sum(1 for o in obs_list if o.evidence_type == "INFERRED")
        nd_cnt = sum(1 for o in obs_list if o.evidence_type == "NOT_DETERMINABLE")

        st.markdown(
            f"""
            <div style="display:flex; gap:10px; margin-top:8px; margin-bottom:12px; flex-wrap:wrap;">
                <span style="background:rgba(2, 132, 199, 0.15); border:1px solid #0284C7; border-radius:6px; padding:4px 10px; font-size:0.8rem; color:#38BDF8;">
                    <b>VISIBLE:</b> {vis_cnt}
                </span>
                <span style="background:rgba(245, 158, 11, 0.15); border:1px solid #F59E0B; border-radius:6px; padding:4px 10px; font-size:0.8rem; color:#FDE68A;">
                    <b>INFERRED:</b> {inf_cnt}
                </span>
                <span style="background:rgba(100, 116, 139, 0.15); border:1px solid #64748B; border-radius:6px; padding:4px 10px; font-size:0.8rem; color:#CBD5E1;">
                    <b>NOT_DETERMINABLE:</b> {nd_cnt}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Filtering controls
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            prio_filter = st.selectbox(
                "Filter by Priority",
                options=["All Priorities", "HIGH ATTENTION", "MEDIUM ATTENTION", "LOW ATTENTION"],
                key="prio_filter_sel",
            )
        with filter_col2:
            present_categories = sorted(list({o.category for o in obs_list}))
            cat_options = ["All Categories"] + present_categories
            cat_filter = st.selectbox(
                "Filter by Category",
                options=cat_options,
                key="cat_filter_sel",
            )

        filtered_obs = obs_list
        if prio_filter != "All Priorities":
            filtered_obs = [o for o in filtered_obs if o.risk_priority == prio_filter]
        if cat_filter != "All Categories":
            filtered_obs = [o for o in filtered_obs if o.category == cat_filter]

        category_icons = {
            "Structural / Concrete": "🏗️",
            "Formwork & Shoring": "📐",
            "Scaffolding": "🪜",
            "Site Safety": "🛡️",
            "PPE": "🦺",
            "Equipment / Machinery": "🚜",
            "Materials": "🧱",
            "Housekeeping / Site Conditions": "🧹",
            "Work Progress": "⏳",
        }

        with st.container(height=520):
            if not filtered_obs:
                st.info("No observations match the selected filter criteria.")
            for idx, obs in enumerate(filtered_obs, 1):
                icon = category_icons.get(obs.category, "📋")
                
                if obs.risk_priority == "HIGH ATTENTION":
                    prio_badge = '<span class="prio-badge prio-high">🔴 Priority: HIGH</span>'
                elif obs.risk_priority == "LOW ATTENTION":
                    prio_badge = '<span class="prio-badge prio-low">🟢 Priority: LOW</span>'
                else:
                    prio_badge = '<span class="prio-badge prio-med">🟡 Priority: MEDIUM</span>'

                if obs.visual_confidence == "HIGH":
                    conf_badge = '<span class="badge-obs">🟢 Confidence: HIGH</span>'
                elif obs.visual_confidence == "REQUIRES_PHYSICAL_VERIFICATION":
                    conf_badge = '<span class="badge-verify">🔍 Confidence: REQUIRES PHYSICAL VERIFICATION</span>'
                else:
                    conf_badge = '<span class="badge-concern">🟡 Confidence: MEDIUM</span>'

                evidence_tag = f"<span style='font-size:0.75rem; color:#94A3B8; margin-left:6px;'>[{obs.evidence_type}]</span>"

                st.markdown(
                    f"""
                    <div class="eng-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
                            <span style="font-weight: 700; font-size: 1.02rem; color: #F1F5F9;">{icon} {obs.category} {evidence_tag}</span>
                            <div style="display: flex; gap: 6px; align-items: center;">
                                {prio_badge}
                                {conf_badge}
                            </div>
                        </div>
                        <div style="font-size: 0.93rem; color: #E2E8F0; margin-bottom: 8px;">
                            <b>Observation:</b> {obs.observation}
                        </div>
                        <div style="font-size: 0.88rem; color: #FDE68A; margin-bottom: 6px;">
                            <b style="color: #FBBF24;">Potential Issue / Risk:</b> {obs.potential_issue}
                        </div>
                        <div style="font-size: 0.88rem; color: #BAE6FD; margin-bottom: 6px;">
                            <b style="color: #60A5FA;">Physical Verification Required:</b> {obs.physical_verification_required}
                        </div>
                        <div style="font-size: 0.88rem; color: #BBF7D0;">
                            <b style="color: #4ADE80;">Recommended Action:</b> {obs.recommended_action}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------------
        # Phase 3C: Local Inspection Report Foundation Preview
        # ---------------------------------------------------------
        curr_id = st.session_state.get("current_inspection_id")
        active_rec = None
        if curr_id:
            active_rec = next(
                (r for r in st.session_state.get("inspection_history", []) if r.inspection_id == curr_id),
                None,
            )
        if not active_rec and report:
            active_rec = InspectionRecord.from_report(report)

        if active_rec:
            st.markdown("---")
            with st.expander("📄 **Inspection Report Preview**", expanded=False):
                st.caption("Clean, structured inspection report representation generated locally from stored data. Zero API calls.")
                rep_repr = generate_inspection_report(active_rec)

                # Report Header Presentation
                st.markdown(
                    f"""
                    <div style="background:#0F172A; border:2px solid #334155; border-radius:10px; padding:18px; margin-bottom:14px;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                            <div>
                                <h3 style="color:#F1F5F9; margin:0 0 4px 0;">{rep_repr.report_title}</h3>
                                <div style="color:#94A3B8; font-size:0.85rem;">Inspection ID: <b style="color:#F59E0B;">{rep_repr.inspection_id}</b></div>
                            </div>
                            <div style="text-align:right;">
                                <div style="color:#94A3B8; font-size:0.85rem;">Date/Time: <b style="color:#CBD5E1;">{rep_repr.timestamp}</b></div>
                                {f'<div style="color:#64748B; font-size:0.75rem;">Source: {rep_repr.source_image_name} ({rep_repr.source_image_size_kb or 0} KB, {rep_repr.source_image_dimensions or "N/A"})</div>' if rep_repr.source_image_name else ''}
                            </div>
                        </div>
                        <hr style="border:none; border-top:1px solid #334155; margin:12px 0;">
                        <div style="font-size:0.9rem; color:#E2E8F0; margin-bottom:6px;">
                            <b style="color:#F59E0B;">Observed Site Activity:</b> {rep_repr.site_activity}
                        </div>
                        <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.5;">
                            <b>Executive Summary:</b> {rep_repr.executive_summary}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Report Observation Summary Counts
                c_total = rep_repr.observation_counts["total"]
                c_high = rep_repr.observation_counts["HIGH ATTENTION"]
                c_med = rep_repr.observation_counts["MEDIUM ATTENTION"]
                c_low = rep_repr.observation_counts["LOW ATTENTION"]

                ev_visible = rep_repr.evidence_counts.get("VISIBLE", 0)
                ev_inferred = rep_repr.evidence_counts.get("INFERRED", 0)
                ev_nd = rep_repr.evidence_counts.get("NOT_DETERMINABLE", 0)

                st.markdown(
                    f"""
                    <div style="display:flex; gap:10px; margin-bottom:8px; flex-wrap:wrap;">
                        <span style="background:#1E293B; border:1px solid #475569; border-radius:6px; padding:4px 10px; font-size:0.82rem; color:#F1F5F9;">Total Observations: <b>{c_total}</b></span>
                        <span style="background:#450A0A; border:1px solid #EF4444; border-radius:6px; padding:4px 10px; font-size:0.82rem; color:#FCA5A5;">🔴 HIGH ATTENTION: <b>{c_high}</b></span>
                        <span style="background:#451A03; border:1px solid #F59E0B; border-radius:6px; padding:4px 10px; font-size:0.82rem; color:#FDE68A;">🟡 MEDIUM ATTENTION: <b>{c_med}</b></span>
                        <span style="background:#064E3B; border:1px solid #10B981; border-radius:6px; padding:4px 10px; font-size:0.82rem; color:#A7F3D0;">🟢 LOW ATTENTION: <b>{c_low}</b></span>
                    </div>
                    <div style="display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap;">
                        <span style="background:rgba(2, 132, 199, 0.15); border:1px solid #0284C7; border-radius:6px; padding:3px 10px; font-size:0.8rem; color:#38BDF8;">VISIBLE: <b>{ev_visible}</b></span>
                        <span style="background:rgba(245, 158, 11, 0.15); border:1px solid #F59E0B; border-radius:6px; padding:3px 10px; font-size:0.8rem; color:#FDE68A;">INFERRED: <b>{ev_inferred}</b></span>
                        <span style="background:rgba(100, 116, 139, 0.15); border:1px solid #64748B; border-radius:6px; padding:3px 10px; font-size:0.8rem; color:#CBD5E1;">NOT_DETERMINABLE: <b>{ev_nd}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Categorized Observation Sections
                if not rep_repr.categorized_observations:
                    st.info("No recorded observations in this inspection.")
                for cat_title, obs_list in rep_repr.categorized_observations.items():
                    st.markdown(f"##### 📌 {cat_title} ({len(obs_list)})")
                    for i, o in enumerate(obs_list, 1):
                        p_color = "#EF4444" if o.risk_priority == "HIGH ATTENTION" else ("#10B981" if o.risk_priority == "LOW ATTENTION" else "#F59E0B")
                        st.markdown(
                            f"""
                            <div style="background:#1E293B; border:1px solid #334155; border-radius:6px; padding:10px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                    <span style="color:#F1F5F9; font-weight:700;">#{i}. [{o.category}]</span>
                                    <span>
                                        <span style="color:{p_color}; font-weight:700;">{o.risk_priority}</span> • 
                                        <span style="color:#94A3B8;">{o.visual_confidence} ({o.evidence_type})</span>
                                    </span>
                                </div>
                                <div style="font-size:0.86rem; color:#E2E8F0; margin-bottom:4px;"><b>Observation:</b> {o.observation}</div>
                                <div style="font-size:0.82rem; color:#FDE68A; margin-bottom:3px;"><b>Potential Issue:</b> {o.potential_issue}</div>
                                <div style="font-size:0.82rem; color:#BAE6FD; margin-bottom:3px;"><b>Physical Verification Required:</b> {o.physical_verification_required}</div>
                                <div style="font-size:0.82rem; color:#BBF7D0;"><b>Recommended Action:</b> {o.recommended_action}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Report Disclaimer Box
                st.markdown(
                    f"""
                    <div style="background:#0F172A; border-left:4px solid #F59E0B; padding:10px 14px; margin-top:14px; margin-bottom:14px; border-radius:0 6px 6px 0; font-size:0.78rem; color:#94A3B8;">
                        <b>Report Notice:</b> {rep_repr.disclaimer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Export Buttons (PDF & CSV - 100% Local / Zero Gemini Calls)
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    try:
                        pdf_bytes = generate_inspection_pdf(rep_repr)
                        pdf_filename = f"CivilVision_Report_{rep_repr.inspection_id}.pdf"
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                            key=f"dl_pdf_{rep_repr.inspection_id}",
                        )
                    except Exception as pdf_err:
                        st.error(f"⚠️ Error generating PDF report: {pdf_err}")
                with dl_col2:
                    try:
                        csv_data = export_inspection_to_csv(active_rec)
                        csv_filename = f"CivilVision_Observations_{rep_repr.inspection_id}.csv"
                        st.download_button(
                            label="📊 Download CSV Data",
                            data=csv_data,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True,
                            key=f"dl_csv_{rep_repr.inspection_id}",
                        )
                    except Exception as csv_err:
                        st.error(f"⚠️ Error generating CSV data: {csv_err}")

    elif st.session_state.get("analysis_result"):
        st.markdown("---")
        st.markdown("### 📋 Initial Visual Inspection Assessment")
        if st.session_state.get("analysis_fallback"):
            st.warning(f"⚠️ {st.session_state.analysis_fallback}")
        with st.container(height=520):
            st.markdown(st.session_state.analysis_result)

# ---------------------------------------------------------
# RIGHT COLUMN: Follow-up Chat & Structured Summary
# ---------------------------------------------------------
with col_right:
    st.markdown("### 💬 Inspection Q&A & Reporting")

    # Quick prompt actions
    st.markdown("##### Quick Follow-up Questions:")
    
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("⚠️ What safety issues are visible?", use_container_width=True):
            st.session_state.pending_prompt = "What safety issues are visible?"
        if st.button("🔍 What should I physically check?", use_container_width=True):
            st.session_state.pending_prompt = "What should I physically check?"
        if st.button("🏗️ What construction activity is happening?", use_container_width=True):
            st.session_state.pending_prompt = "What construction activity is happening?"
    with q_col2:
        if st.button("❓ What cannot be determined from this image?", use_container_width=True):
            st.session_state.pending_prompt = "What information cannot be determined from this image?"
        if st.button("📝 Create an inspection summary", use_container_width=True):
            st.session_state.pending_prompt = "Create an inspection summary."
        
    st.markdown("---")

    # Inspection Summary Generator Button
    summary_btn_col1, summary_btn_col2 = st.columns([1.5, 1])
    with summary_btn_col1:
        gen_summary = st.button("📑 Generate Inspection Summary", type="secondary", use_container_width=True)
    
    if gen_summary:
        if not st.session_state.image_bytes:
            st.warning("⚠️ Please upload and analyze an image first before generating a summary.")
        else:
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Google Gemini API Key required.")
            else:
                with st.spinner("Synthesizing structured inspection summary report..."):
                    try:
                        from google import genai
                        from google.genai import types

                        client = init_genai_client(api_key)
                        if client:
                            image_part = types.Part.from_bytes(
                                data=st.session_state.image_bytes,
                                mime_type=st.session_state.image_mime,
                            )
                            
                            # Clean context strictly containing visual analysis text (no fallback notices)
                            clean_analysis_context = ""
                            if st.session_state.analysis_result:
                                clean_analysis_context = f"\n\nContext from initial visual inspection:\n{st.session_state.analysis_result}"

                            summary_prompt = f"{SUMMARY_GENERATION_PROMPT}{clean_analysis_context}"

                            response_text, used_fallback, fallback_notice = generate_content_with_retry_and_fallback(
                                client=client,
                                model=selected_model,
                                contents=[image_part, summary_prompt],
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_INSTRUCTION,
                                    temperature=0.2,
                                ),
                                api_key=api_key,
                            )
                            st.session_state.inspection_summary = response_text
                            st.session_state.summary_fallback = fallback_notice if used_fallback else None
                            st.success("✅ Formal Inspection Summary generated!")
                    except Exception as e:
                        err_msg = str(e)
                        if "10053" in err_msg or "connection" in err_msg.lower():
                            st.error("⚠️ Connection interrupted by local host network ([WinError 10053]). Please retry summary generation.")
                        elif "temporarily unavailable" in err_msg.lower() or "high demand" in err_msg.lower():
                            st.error("⚠️ Gemini services are currently experiencing high demand. Please wait a few moments and retry.")
                        elif "api key" in err_msg.lower() or "unauthenticated" in err_msg.lower():
                            st.error("❌ Invalid or missing Google Gemini API key. Please check your key in the sidebar.")
                        else:
                            st.error(f"⚠️ Error generating summary: {e}")

    # Display Inspection Summary if available
    if st.session_state.inspection_summary:
        with st.expander("📄 **Generated Inspection Summary Report**", expanded=True):
            # Single UI-only fallback notice rendering
            if st.session_state.get("summary_fallback"):
                st.warning(f"⚠️ {st.session_state.summary_fallback}")
            st.markdown(st.session_state.inspection_summary)
            
            # Download button for summary
            st.download_button(
                label="📥 Download civilvision_inspection_summary.txt",
                data=st.session_state.inspection_summary,
                file_name="civilvision_inspection_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ---------------------------------------------------------
    # Conversational Chat Area
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("##### 🗨️ Conversational Follow-up")

    # Single UI-only fallback notification for chat if applicable
    if st.session_state.get("last_chat_fallback"):
        st.warning(f"⚠️ {st.session_state.last_chat_fallback}")

    # Chat message container
    chat_container = st.container(height=350)
    with chat_container:
        if not st.session_state.chat_history:
            st.caption("No conversation yet. Upload an image and ask questions above.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Handle chat input or pending quick prompt
    user_input = st.chat_input("Ask a question about visible site conditions, PPE, scaffolding, or rebar...")
    prompt_to_process = None

    if st.session_state.pending_prompt:
        prompt_to_process = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif user_input:
        prompt_to_process = user_input

    if prompt_to_process:
        if not st.session_state.image_bytes:
            st.warning("⚠️ Please upload a construction site photograph first.")
        else:
            api_key = get_api_key()
            if not api_key:
                st.error("❌ Google Gemini API Key required.")
            else:
                # Add user message to state
                st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
                
                # Call Gemini with image and conversation context
                with st.spinner("CivilVision AI is thinking..."):
                    try:
                        from google import genai
                        from google.genai import types

                        client = init_genai_client(api_key)
                        if client:
                            image_part = types.Part.from_bytes(
                                data=st.session_state.image_bytes,
                                mime_type=st.session_state.image_mime,
                            )
                            
                            # Construct context-rich prompt
                            chat_wrapper = build_chat_prompt(
                                prompt_to_process,
                                has_prior_analysis=bool(st.session_state.analysis_result),
                            )
                            
                            # Include previous conversation snippets for multi-turn awareness (pure user/assistant text)
                            history_context = ""
                            qa_history = [
                                m for m in st.session_state.chat_history[:-1]
                                if not (m["role"] == "assistant" and "Visual Inspection Assessment Completed" in m["content"])
                            ]
                            if qa_history:
                                history_context = "\n\nRecent Conversation History:\n"
                                for prev in qa_history[-6:]:
                                    history_context += f"- {prev['role'].capitalize()}: {prev['content']}\n"

                            full_prompt = f"{chat_wrapper}\n{history_context}"

                            response_text, used_fallback, fallback_notice = generate_content_with_retry_and_fallback(
                                client=client,
                                model=selected_model,
                                contents=[image_part, full_prompt],
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_INSTRUCTION,
                                    temperature=0.2,
                                ),
                                api_key=api_key,
                            )
                            
                            # Append assistant message (pure model text, NEVER injected with fallback notice)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response_text}
                            )
                            if used_fallback:
                                st.session_state.last_chat_fallback = fallback_notice
                            else:
                                st.session_state.last_chat_fallback = None
                            st.rerun()
                    except Exception as e:
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                            st.session_state.chat_history.pop()
                        err_msg = str(e)
                        if "10053" in err_msg or "connection" in err_msg.lower():
                            st.error("⚠️ Connection interrupted by local host network ([WinError 10053]). Please try submitting your question again.")
                        elif "temporarily unavailable" in err_msg.lower() or "high demand" in err_msg.lower():
                            st.error("⚠️ Gemini services (Primary 3.7 Flash and Fallback 3.6 Flash) are currently experiencing high demand. Please wait a few moments and retry.")
                        elif "api key" in err_msg.lower() or "unauthenticated" in err_msg.lower():
                            st.error("❌ Invalid or missing Google Gemini API key. Please check your key in the sidebar.")
                        else:
                            st.error(f"⚠️ Chat generation error: {e}")

# ---------------------------------------------------------
# TAB 2: Phase 3B Inspection Comparison (Pure Local / Zero Gemini Calls)
# ---------------------------------------------------------
with tab_compare:
    st.markdown("### 📊 Inspection Comparison")
    st.caption("Compare two completed visual inspections from this session side-by-side using local evaluation. Zero API calls.")

    history_list = st.session_state.get("inspection_history", [])

    if len(history_list) < 2:
        st.info("ℹ️ At least two inspections are required for comparison. Please upload and inspect additional images in the **Current Site Inspection** tab.")
    else:
        # Selection controls
        options_map = {
            f"{r.inspection_id} ({r.timestamp}) — {r.site_activity[:40]}...": r.inspection_id
            for r in history_list
        }
        labels = list(options_map.keys())

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            sel_label_a = st.selectbox(
                "Inspection A (Baseline)",
                options=labels,
                index=0,
                key="sel_cmp_a",
            )
        with cmp_col2:
            default_b_idx = 1 if len(labels) > 1 else 0
            sel_label_b = st.selectbox(
                "Inspection B (Comparison)",
                options=labels,
                index=default_b_idx,
                key="sel_cmp_b",
            )

        id_a = options_map[sel_label_a]
        id_b = options_map[sel_label_b]

        if id_a == id_b:
            st.warning("⚠️ Please select two different inspections to compare. Inspection A and Inspection B cannot be identical.")
        else:
            rec_a = next(r for r in history_list if r.inspection_id == id_a)
            rec_b = next(r for r in history_list if r.inspection_id == id_b)

            # Local comparison calculation (zero Gemini calls)
            cmp_res = compare_inspection_records(rec_a, rec_b)

            # 1. Summary of Changes (Factual, local, non-subjective)
            st.markdown("#### 📝 Factual Change Summary")
            with st.container():
                for stmt in cmp_res.change_summary_statements:
                    st.markdown(f"- {stmt}")

            st.markdown("---")

            # 2. Side-by-Side Overview
            st.markdown("#### 🔍 Inspection Overview & Numerical Differences")
            st.caption("Differences calculated as: Inspection B − Inspection A")

            m1, m2, m3, m4 = st.columns(4)
            diff_obs_str = f"{cmp_res.total_obs_diff:+d}" if cmp_res.total_obs_diff != 0 else "0"
            m1.metric("Total Observations", f"{cmp_res.total_obs_b}", delta=diff_obs_str)

            high_diff = cmp_res.priority_diffs["HIGH ATTENTION"]
            high_diff_str = f"{high_diff:+d}" if high_diff != 0 else "0"
            m2.metric("🔴 High Attention", f"{rec_b.high_attention_count}", delta=high_diff_str, delta_color="inverse")

            med_diff = cmp_res.priority_diffs["MEDIUM ATTENTION"]
            med_diff_str = f"{med_diff:+d}" if med_diff != 0 else "0"
            m3.metric("🟡 Medium Attention", f"{rec_b.medium_attention_count}", delta=med_diff_str, delta_color="inverse")

            low_diff = cmp_res.priority_diffs["LOW ATTENTION"]
            low_diff_str = f"{low_diff:+d}" if low_diff != 0 else "0"
            m4.metric("🟢 Low Attention", f"{rec_b.low_attention_count}", delta=low_diff_str)

            # Side-by-side details table/cards
            ov_col_a, ov_col_b = st.columns(2)
            with ov_col_a:
                st.markdown(
                    f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:14px; margin-top:8px;">
                        <h4 style="color:#60A5FA; margin:0 0 6px 0;">Inspection A</h4>
                        <div style="font-size:0.85rem; color:#F59E0B; font-weight:700;">{rec_a.inspection_id}</div>
                        <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:8px;">🕒 {rec_a.timestamp}</div>
                        <div style="font-size:0.85rem; color:#E2E8F0;"><b>Activity:</b> {rec_a.site_activity}</div>
                        <div style="font-size:0.82rem; color:#CBD5E1; margin-top:6px;"><b>Executive Summary:</b> {rec_a.executive_summary}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with ov_col_b:
                st.markdown(
                    f"""
                    <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:14px; margin-top:8px;">
                        <h4 style="color:#34D399; margin:0 0 6px 0;">Inspection B</h4>
                        <div style="font-size:0.85rem; color:#F59E0B; font-weight:700;">{rec_b.inspection_id}</div>
                        <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:8px;">🕒 {rec_b.timestamp}</div>
                        <div style="font-size:0.85rem; color:#E2E8F0;"><b>Activity:</b> {rec_b.site_activity}</div>
                        <div style="font-size:0.82rem; color:#CBD5E1; margin-top:6px;"><b>Executive Summary:</b> {rec_b.executive_summary}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # 3. Category & Confidence Breakdown
            breakdown_col1, breakdown_col2 = st.columns(2)
            with breakdown_col1:
                st.markdown("#### 📐 Category Breakdown")
                cat_rows = []
                for cat, diff in cmp_res.category_diffs.items():
                    cnt_a = cmp_res.category_counts_a[cat]
                    cnt_b = cmp_res.category_counts_b[cat]
                    diff_sign = f"{diff:+d}" if diff != 0 else "0"
                    status = "increased" if diff > 0 else ("decreased" if diff < 0 else "same")
                    cat_rows.append({
                        "Category": cat,
                        "Insp A": cnt_a,
                        "Insp B": cnt_b,
                        "Difference (B - A)": diff_sign,
                        "Status": status,
                    })
                st.dataframe(cat_rows, use_container_width=True, hide_index=True)

            with breakdown_col2:
                st.markdown("#### 🎯 Visual Confidence Breakdown")
                conf_rows = []
                for lvl, diff in cmp_res.confidence_diffs.items():
                    cnt_a = cmp_res.confidence_counts_a[lvl]
                    cnt_b = cmp_res.confidence_counts_b[lvl]
                    diff_sign = f"{diff:+d}" if diff != 0 else "0"
                    status = "increased" if diff > 0 else ("decreased" if diff < 0 else "same")
                    conf_rows.append({
                        "Confidence Level": lvl,
                        "Insp A": cnt_a,
                        "Insp B": cnt_b,
                        "Difference (B - A)": diff_sign,
                        "Status": status,
                    })
                st.dataframe(conf_rows, use_container_width=True, hide_index=True)

            st.markdown("---")

            # 4. Observation-Level Side-by-Side Detail
            st.markdown("#### 📋 Observation-Level Detail")
            st.caption("Observations grouped by category. Evaluated strictly from stored structured observations without semantic assumptions.")

            all_cats = sorted(list(
                {o.category for o in rec_a.observations}.union(
                    {o.category for o in rec_b.observations}
                )
            ))

            for cat in all_cats:
                obs_a = [o for o in rec_a.observations if o.category == cat]
                obs_b = [o for o in rec_b.observations if o.category == cat]

                with st.expander(f"📁 **{cat}** (Insp A: {len(obs_a)} | Insp B: {len(obs_b)})", expanded=True):
                    col_obs_a, col_obs_b = st.columns(2)
                    with col_obs_a:
                        st.markdown(f"**Inspection A Observations ({len(obs_a)}):**")
                        if not obs_a:
                            st.caption("No observations in this category.")
                        for idx, o in enumerate(obs_a, 1):
                            prio_color = "#EF4444" if o.risk_priority == "HIGH ATTENTION" else ("#10B981" if o.risk_priority == "LOW ATTENTION" else "#F59E0B")
                            st.markdown(
                                f"""
                                <div style="background:#0F172A; border:1px solid #334155; border-radius:6px; padding:10px; margin-bottom:8px;">
                                    <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                                        <span style="color:{prio_color}; font-weight:700;">{o.risk_priority}</span>
                                        <span style="color:#94A3B8;">[{o.visual_confidence}]</span>
                                    </div>
                                    <div style="font-size:0.85rem; color:#F1F5F9; margin-top:4px;"><b>Observation:</b> {o.observation}</div>
                                    <div style="font-size:0.8rem; color:#FDE68A; margin-top:2px;"><b>Issue / Risk:</b> {o.potential_issue}</div>
                                    <div style="font-size:0.8rem; color:#BAE6FD; margin-top:2px;"><b>Verification:</b> {o.physical_verification_required}</div>
                                    <div style="font-size:0.8rem; color:#BBF7D0; margin-top:2px;"><b>Action:</b> {o.recommended_action}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    with col_obs_b:
                        st.markdown(f"**Inspection B Observations ({len(obs_b)}):**")
                        if not obs_b:
                            st.caption("No observations in this category.")
                        for idx, o in enumerate(obs_b, 1):
                            prio_color = "#EF4444" if o.risk_priority == "HIGH ATTENTION" else ("#10B981" if o.risk_priority == "LOW ATTENTION" else "#F59E0B")
                            st.markdown(
                                f"""
                                <div style="background:#0F172A; border:1px solid #334155; border-radius:6px; padding:10px; margin-bottom:8px;">
                                    <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                                        <span style="color:{prio_color}; font-weight:700;">{o.risk_priority}</span>
                                        <span style="color:#94A3B8;">[{o.visual_confidence}]</span>
                                    </div>
                                    <div style="font-size:0.85rem; color:#F1F5F9; margin-top:4px;"><b>Observation:</b> {o.observation}</div>
                                    <div style="font-size:0.8rem; color:#FDE68A; margin-top:2px;"><b>Issue / Risk:</b> {o.potential_issue}</div>
                                    <div style="font-size:0.8rem; color:#BAE6FD; margin-top:2px;"><b>Verification:</b> {o.physical_verification_required}</div>
                                    <div style="font-size:0.8rem; color:#BBF7D0; margin-top:2px;"><b>Action:</b> {o.recommended_action}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            st.markdown(
                """
                <div style="font-size:0.78rem; color:#94A3B8; background:#1E293B; border-radius:6px; padding:10px; margin-top:14px;">
                    <b>Engineering Comparison Notice:</b> Visual comparison compares recorded visual observations across photographic inspections. It does not establish structural adequacy, code compliance, concrete strength, or overall site safety. Mandatory physical engineering inspections and testing remain required.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 10px 0;">
        <b>CivilVision AI</b> • AI-Powered Construction Site Visual Inspection & Reporting Assistant<br>
        <i>Disclaimer: This software is an educational and preliminary visual screening tool. Visual AI does not replace licensed professional structural engineering inspections, physical non-destructive testing (NDT), or site safety compliance sign-offs.</i>
    </div>
    """,
    unsafe_allow_html=True,
)
