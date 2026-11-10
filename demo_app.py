
import streamlit as st
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
# Conversation State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

st.title("📄 AI Research Paper Auditor")
st.caption(
    "Understand, analyze and evaluate research papers "
    "using specialized AI agents."
)

st.divider()

# -----------------------------
# PDF Upload
# -----------------------------

st.subheader("1. Upload Research Paper")

uploaded_pdf = st.file_uploader(
    "Choose your research paper",
    type=["pdf"],
)

if uploaded_pdf:
    st.success(f"Uploaded: {uploaded_pdf.name}")
    st.session_state.pdf_bytes = uploaded_pdf.getvalue()
    st.session_state.pdf_name = uploaded_pdf.name
# -----------------------------
# Conversation History
# -----------------------------

st.subheader("Conversation")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Question Input
# -----------------------------

st.subheader("2. Ask Your Question")

input_method = st.radio(
    "How would you like to ask?",
    ["Speak", "Type"],
    horizontal=True,
)

user_question = ""
audio_recording = None
language_code = "hi-IN"

if input_method == "Speak":

    st.write("🎙️ **Speak your research question**")

    language_choice = st.selectbox(
        "Spoken language",
        [
            "Hindi / Hinglish",
            "English",
        ],
    )

    language_code = (
        "hi-IN"
        if language_choice == "Hindi / Hinglish"
        else "en-IN"
    )

    audio_recording = st.audio_input(
        "Record your question"
    )

    if audio_recording:
        st.audio(audio_recording)

else:

    user_question = st.text_area(
        "What would you like to understand about this paper?",
        placeholder=(
            "Example: Explain how the proposed antenna "
            "design achieves isolation."
        ),
        height=120,
    )


# -----------------------------
# Processing
# -----------------------------

if st.button(
    "Analyze Research Paper",
    type="primary",
    use_container_width=True,
):

    if uploaded_pdf is None:
        st.error("Please upload a research paper PDF.")
        st.stop()

    temp_pdf_path = None

    try:

        # Stage 0: Obtain question from voice or text

        if input_method == "Speak":

            if audio_recording is None:
                st.error("Please record your question first.")
                st.stop()

            with st.spinner("Gnani Prisma is transcribing your voice..."):

                transcript = transcribe_audio(
                    audio_bytes=audio_recording.getvalue(),
                    filename="recording.wav",
                    language_code=language_code,
                )

            st.subheader("3. Voice Transcript")

            st.write(transcript)

            original_question = transcript

        else:

            if not user_question.strip():
                st.error("Please enter your question.")
                st.stop()

            original_question = user_question

        # Stage 1: Groq normalization

        with st.spinner("Groq is understanding your question..."):

            normalized = normalize_user_query(
                original_question
            )

        st.subheader("4. Query Understanding")

        st.write("**Original input:**")
        st.write(original_question)

        st.write("**Normalized question:**")
        st.write(normalized["normalized_question"])

        st.write("**Understood intent:**")
        st.write(normalized["intent_summary"])

        if normalized["needs_clarification"]:

            st.warning(
                normalized["clarification_question"]
            )

            st.stop()

        question = normalized["normalized_question"]
        intent_summary = normalized["intent_summary"]
        st.session_state.messages.append({
       "role": "user",
       "content": original_question,
        })

        # Save uploaded PDF temporarily

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_file.write(uploaded_pdf.getvalue())
            temp_pdf_path = Path(temp_file.name)

        # Stage 2: Load agents and route query

        with st.spinner(
            "Question Answerer is selecting a specialist..."
        ):

            agents = build_agents()

            (
                selected_key,
                reason,
                selected_agent,
                route_raw,
            ) = route_query(
                question,
                intent_summary,
                agents,
            )

        st.subheader("5. Specialist Selection")

        st.success(
            f"Selected specialist: {selected_key}"
        )

        st.write("**Why this specialist?**")
        st.write(reason)

        with st.expander("View routing output"):
            st.code(route_raw, language="json")

        # Stage 3: Execute selected specialist

        with st.spinner(
            f"{selected_key} is analyzing your research paper..."
        ):

            answer = execute_specialist(
                selected_agent,
                temp_pdf_path,
                question,
            )

        answer_text = str(
            getattr(answer, "raw", answer)
        )
        

        st.divider()

        st.subheader("6. Research Paper Analysis")

        st.markdown(answer_text)

        # Save output

        output_file = save_outputs(
            route_raw,
            selected_key,
            reason,
            answer,
        )

        st.success("Analysis completed successfully.")

        st.caption(f"Saved output: {output_file}")

        st.download_button(
            label="Download Analysis",
            data=answer_text,
            file_name="research_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except Exception as error:

        st.error(f"Analysis failed: {error}")

    finally:

        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()


st.divider()

st.caption(
    "Powered by Gnani Prisma v3.0, Groq and CrewAI."
)
