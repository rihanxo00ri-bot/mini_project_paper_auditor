import streamlit as st
import streamlit.components.v1 as components
import tempfile
from pathlib import Path

from intent_normalizer import normalize_user_query
from integrations.prisma_stt import transcribe_audio

from paper_auditor import (
    build_agents,
    route_query,
    execute_specialist,
    save_outputs,
)


st.set_page_config(
    page_title="AI Research Paper Auditor",
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Fixed Bottom Input Bar (CSS)
# -----------------------------

st.markdown(
    """
    <style>
    /* Reserve space at bottom so chat isn't hidden behind input bar */
    .block-container {
        padding-bottom: 220px !important;
    }

    /* The container that holds the input bar */
    div[data-testid="stVerticalBlock"] > div:has(> div > div[data-testid="stHorizontalBlock"].fixed-input-anchor),
    .fixed-input-wrapper {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: var(--background-color, #ffffff);
        border-top: 1px solid rgba(128,128,128,0.25);
        padding: 12px 24px 16px 24px;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.06);
    }

    /* Pull the fixed input to be centered like main content */
    .fixed-input-inner {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pending_submit" not in st.session_state:
    st.session_state.pending_submit = False

@st.cache_resource(show_spinner=False)
def get_agents():
    return build_agents()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.subheader("📎 Research Paper")
    uploaded_pdf = st.file_uploader("Upload / replace PDF", type=["pdf"])
    if uploaded_pdf:
        st.session_state.pdf_bytes = uploaded_pdf.getvalue()
        st.session_state.pdf_name = uploaded_pdf.name
        st.success(f"Loaded: {uploaded_pdf.name}")

    if st.session_state.pdf_name:
        st.info(f"Active paper: **{st.session_state.pdf_name}**")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Header
# -----------------------------

st.title("📄 AI Research Paper Auditor")
st.caption(
    "Understand, analyze and evaluate research papers "
    "using specialized AI agents. Ask follow-up questions freely."
)
st.divider()


# -----------------------------
# Conversation history
# -----------------------------

st.subheader("💬 Conversation")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("meta"):
            with st.expander("🔍 How this answer was produced"):
                st.markdown(message["meta"])


# -----------------------------
# Auto-scroll to bottom after each turn
# -----------------------------

components.html(
    """
    <script>
    const doc = window.parent.document;
    const main = doc.querySelector('section.main') || doc.querySelector('[data-testid="stAppViewContainer"]');
    if (main) { main.scrollTop = main.scrollHeight; }
    window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
    </script>
    """,
    height=0,
)


# -----------------------------
# Fixed bottom input bar
# -----------------------------

# This container is styled fixed via CSS class
input_bar = st.container()
with input_bar:
    st.markdown('<div class="fixed-input-wrapper"><div class="fixed-input-inner">', unsafe_allow_html=True)

    input_method = st.radio(
        "Input method",
        ["Type", "Speak"],
        horizontal=True,
        label_visibility="collapsed",
    )

    user_question = ""
    audio_recording = None
    language_code = "hi-IN"

    if input_method == "Speak":
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            language_choice = st.selectbox(
                "Spoken language",
                ["Hindi / Hinglish", "English"],
                label_visibility="collapsed",
            )
            language_code = (
                "hi-IN" if language_choice == "Hindi / Hinglish" else "en-IN"
            )
        with col2:
            audio_recording = st.audio_input(
                "Record your question",
                label_visibility="collapsed",
            )
        with col3:
            st.write("")
            st.write("")
            submit = st.button(
                "Send 🎙️",
                type="primary",
                use_container_width=True,
                key="submit_voice",
            )
    else:
        col1, col2 = st.columns([6, 1])
        with col1:
            user_question = st.text_input(
                "Ask a question",
                placeholder="Ask anything about this paper... (press Enter to send)",
                label_visibility="collapsed",
            )
        with col2:
            submit = st.button(
                "Send ➤",
                type="primary",
                use_container_width=True,
                key="submit_text",
            )

    st.markdown("</div></div>", unsafe_allow_html=True)


# -----------------------------
# Processing (per turn)
# -----------------------------

# Trigger either by button click OR Enter in text input
text_enter_pressed = input_method == "Type" and user_question.strip() != ""
should_process = submit or text_enter_pressed

if should_process:

    if not st.session_state.pdf_bytes:
        st.error("Please upload a research paper PDF first (in the sidebar).")
        st.stop()

    temp_pdf_path = None

    try:
        # -------- Stage 0: Obtain question --------
        if input_method == "Speak":
            if audio_recording is None:
                st.error("Please record your question first.")
                st.stop()
            with st.spinner("Transcribing your voice..."):
                transcript = transcribe_audio(
                    audio_bytes=audio_recording.getvalue(),
                    filename="recording.wav",
                    language_code=language_code,
                )
            original_question = transcript
        else:
            original_question = user_question

        # -------- Show user turn --------
        st.session_state.messages.append(
            {"role": "user", "content": original_question}
        )

        # -------- Assistant turn --------
        meta_lines = []

        with st.spinner("Understanding your question..."):
            normalized = normalize_user_query(original_question)

        meta_lines.append("**Original input:**")
        meta_lines.append(original_question)
        meta_lines.append("")
        meta_lines.append("**Normalized question:**")
        meta_lines.append(normalized["normalized_question"])
        meta_lines.append("")
        meta_lines.append("**Understood intent:**")
        meta_lines.append(normalized["intent_summary"])

        if normalized["needs_clarification"]:
            clarification = normalized["clarification_question"]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"🤔 {clarification}",
                    "meta": "\n".join(meta_lines),
                }
            )
            st.rerun()

        question = normalized["normalized_question"]
        intent_summary = normalized["intent_summary"]

        # Temp PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(st.session_state.pdf_bytes)
            temp_pdf_path = Path(temp_file.name)

        # Route
        with st.spinner("Selecting specialist..."):
            agents = get_agents()
            (
                selected_key,
                reason,
                selected_agent,
                route_raw,
            ) = route_query(question, intent_summary, agents)

        meta_lines.append("")
        meta_lines.append(f"**Specialist:** `{selected_key}`")
        meta_lines.append("")
        meta_lines.append("**Why this specialist?**")
        meta_lines.append(reason)

        # Execute
        with st.spinner(f"{selected_key} is analyzing your research paper..."):
            answer = execute_specialist(selected_agent, temp_pdf_path, question)

        answer_text = str(getattr(answer, "raw", answer))

        # Save output (best-effort)
        try:
            save_outputs(route_raw, selected_key, reason, answer)
        except Exception as save_err:
            st.session_state.setdefault("save_warnings", []).append(str(save_err))

        # Persist assistant turn
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "meta": "\n".join(meta_lines),
                "route_raw": route_raw,
            }
        )

        st.rerun()

    except Exception as error:
        st.error(f"Analysis failed: {error}")

    finally:
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()


st.caption("Powered by Gnani Prisma v3.0, Groq and CrewAI.")