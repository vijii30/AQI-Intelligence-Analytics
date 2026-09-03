
# ============================================================
# AQI INTELLIGENCE DASHBOARD
# ============================================================

import os
import json
import time
import base64
import io
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path("/content/AQI_PROJECT")

OUTPUT_DIR = PROJECT / "outputs"
RAG_DIR = PROJECT / "rag"
MODEL_DIR = PROJECT / "models"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AQI Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
"""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    color: #667085;
    font-size: 16px;
    margin-bottom: 25px;
}

.ai-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.voice-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# GEMINI API
# ============================================================

API_KEY = os.environ.get(
    "GOOGLE_API_KEY",
    ""
).strip()


if API_KEY:

    client = genai.Client(
        api_key=API_KEY
    )

else:

    client = None


# ============================================================
# GEMINI MODELS
# ============================================================

TEXT_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite"
]

AUDIO_MODEL = "gemini-3.7-flash"

# Official Gemini TTS model
TTS_MODEL = "gemini-3.1-flash-tts-preview"


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversation_memory" not in st.session_state:

    st.session_state.conversation_memory = []


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

@st.cache_data
def load_dashboard_data():

    path = (
        OUTPUT_DIR /
        "Final_AQI_Dashboard_Data.csv"
    )

    if not path.exists():

        return pd.DataFrame()

    data = pd.read_csv(path)

    data.columns = [

        str(c)
        .strip()
        .lower()
        .replace(" ", "_")

        for c in data.columns

    ]

    return data


df = load_dashboard_data()


# ============================================================
# LOAD RAG
# ============================================================

@st.cache_resource
def load_rag():

    documents_path = (
        RAG_DIR /
        "rag_documents.json"
    )

    vectorizer_path = (
        RAG_DIR /
        "tfidf_vectorizer.pkl"
    )

    matrix_path = (
        RAG_DIR /
        "rag_matrix.npz"
    )


    if not (
        documents_path.exists()
        and vectorizer_path.exists()
        and matrix_path.exists()
    ):

        return None, None, None


    with open(
        documents_path,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)


    vectorizer = joblib.load(
        vectorizer_path
    )


    from scipy.sparse import load_npz

    matrix = load_npz(
        matrix_path
    )


    return (
        documents,
        vectorizer,
        matrix
    )


rag_documents, vectorizer, rag_matrix = load_rag()


# ============================================================
# AQI CATEGORY
# ============================================================

def aqi_category(value):

    try:

        value = float(value)

    except:

        return "Unknown"


    if value <= 50:

        return "Good"

    elif value <= 100:

        return "Satisfactory"

    elif value <= 200:

        return "Moderate"

    elif value <= 300:

        return "Poor"

    elif value <= 400:

        return "Very Poor"

    else:

        return "Severe"


# ============================================================
# FIND AQI COLUMN
# ============================================================

def get_aqi_column(data):

    if data.empty:

        return None


    candidates = [

        c

        for c in data.columns

        if (
            "aqi" in c.lower()
            and "predicted" not in c.lower()
        )

    ]


    if candidates:

        return candidates[0]


    return None


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    top_k=5
):

    if (
        rag_documents is None
        or vectorizer is None
        or rag_matrix is None
    ):

        return []


    try:

        query_vector = vectorizer.transform(
            [query]
        )


        scores = cosine_similarity(
            query_vector,
            rag_matrix
        )[0]


        top_indices = np.argsort(
            scores
        )[::-1][:top_k]


        results = []


        for index in top_indices:

            results.append({

                "text":
                    rag_documents[index].get(
                        "text",
                        ""
                    ),

                "source":
                    rag_documents[index].get(
                        "source",
                        "AQI Dataset"
                    ),

                "score":
                    float(
                        scores[index]
                    )

            })


        return results


    except Exception as e:

        print(
            "RAG retrieval error:",
            e
        )

        return []


# ============================================================
# GEMINI TEXT GENERATION
# ============================================================

def call_gemini(prompt):

    if client is None:

        return (
            "Gemini API key is not configured."
        )


    last_error = None


    for model in TEXT_MODELS:

        for attempt in range(3):

            try:

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                )


                if (
                    response
                    and response.text
                ):

                    return response.text.strip()


            except Exception as e:

                last_error = e

                error_text = str(e)

                print(
                    f"Gemini error | "
                    f"{model} | "
                    f"attempt {attempt + 1}: "
                    f"{error_text}"
                )


                # ------------------------------------------------
                # 503
                # ------------------------------------------------

                if (
                    "503" in error_text
                    or
                    "UNAVAILABLE"
                    in error_text
                ):

                    if attempt < 2:

                        time.sleep(
                            2 ** attempt
                        )

                        continue

                    break


                # ------------------------------------------------
                # 429
                # ------------------------------------------------

                if (
                    "429" in error_text
                    or
                    "RESOURCE_EXHAUSTED"
                    in error_text
                ):

                    if attempt < 2:

                        time.sleep(
                            3 * (attempt + 1)
                        )

                        continue

                    break


                break


    print(
        "Final Gemini error:",
        last_error
    )


    return (
        "Gemini is temporarily unavailable. "
        "Please try again."
    )


# ============================================================
# SPEECH → TEXT
# ============================================================

def transcribe_voice(
    audio_bytes
):

    if client is None:

        return None


    if not audio_bytes:

        return None


    try:

        encoded_audio = base64.b64encode(
            audio_bytes
        ).decode("utf-8")


        response = (
            client.models.generate_content(

                model=AUDIO_MODEL,

                contents=[

                    {

                        "parts": [

                            {

                                "text":
                                """
You are a speech recognition
assistant for an AQI dashboard.

Listen to the user's audio.

Convert ONLY the spoken question
into clean text.

Do NOT answer the question.

Do NOT explain the AQI.

Do NOT add information.

Return ONLY the question.

Example:

User says:
What is the AQI of Chennai?

Return exactly:
What is the AQI of Chennai?
"""
                            },

                            {

                                "inline_data": {

                                    "mime_type":
                                        "audio/wav",

                                    "data":
                                        encoded_audio

                                }

                            }

                        ]

                    }

                ]

            )
        )


        if (
            response
            and response.text
        ):

            text = response.text.strip()

            text = text.strip(
                '"'
            ).strip(
                "'"
            )

            return text


    except Exception as e:

        print(
            "Speech recognition error:",
            e
        )


    return None


# ============================================================
# PCM → WAV CONVERTER
# ============================================================
#
# THIS IS THE IMPORTANT FIX.
#
# Gemini TTS returns raw PCM audio.
#
# Browser audio players need a valid audio
# container such as WAV.
#
# Therefore:
#
# PCM → WAV
#
# ============================================================

def pcm_to_wav(
    pcm_data,
    sample_rate=24000,
    channels=1,
    sample_width=2
):

    if not pcm_data:

        return None


    try:

        wav_buffer = io.BytesIO()


        with wave.open(
            wav_buffer,
            "wb"
        ) as wav_file:

            wav_file.setnchannels(
                channels
            )

            wav_file.setsampwidth(
                sample_width
            )

            wav_file.setframerate(
                sample_rate
            )

            wav_file.writeframes(
                pcm_data
            )


        return wav_buffer.getvalue()


    except Exception as e:

        print(
            "PCM → WAV conversion error:",
            e
        )

        return None


# ============================================================
# GEMINI TEXT → SPEECH
# ============================================================
#
# FIXED TTS IMPLEMENTATION
#
# Gemini:
#
# Text
#  ↓
# PCM audio
#  ↓
# WAV conversion
#  ↓
# Streamlit audio player
#
# ============================================================

def generate_gemini_speech(
    answer
):

    if client is None:

        return None


    if not answer:

        return None


    try:

        print(
            "🔊 Generating Gemini TTS..."
        )


        response = (
            client.models.generate_content(

                model=TTS_MODEL,

                contents=(
                    "Speak naturally and clearly "
                    "as an AQI assistant. "
                    "Explain the following answer "
                    "in a friendly human voice:\n\n"
                    + answer
                ),

                config=types.GenerateContentConfig(

                    response_modalities=[
                        "AUDIO"
                    ],

                    speech_config=types.SpeechConfig(

                        voice_config=
                        types.VoiceConfig(

                            prebuilt_voice_config=
                            types.PrebuiltVoiceConfig(

                                voice_name="Kore"

                            )

                        )

                    )

                )

            )
        )


        # ====================================================
        # FIND AUDIO DATA
        # ====================================================

        pcm_data = None


        if (
            response
            and response.candidates
        ):

            for candidate in response.candidates:

                if not candidate.content:

                    continue


                for part in candidate.content.parts:

                    if (
                        hasattr(
                            part,
                            "inline_data"
                        )
                        and
                        part.inline_data
                    ):

                        pcm_data = (
                            part.inline_data.data
                        )

                        break


                if pcm_data:

                    break


        # ====================================================
        # CHECK AUDIO
        # ====================================================

        if not pcm_data:

            print(
                "❌ Gemini returned no audio data."
            )

            return None


        print(
            "✅ Gemini PCM audio received."
        )


        # ====================================================
        # HANDLE BASE64 IF NECESSARY
        # ====================================================

        if isinstance(
            pcm_data,
            str
        ):

            try:

                pcm_data = base64.b64decode(
                    pcm_data
                )

            except Exception as e:

                print(
                    "Base64 decode error:",
                    e
                )

                return None


        # ====================================================
        # CONVERT PCM → WAV
        # ====================================================

        wav_data = pcm_to_wav(

            pcm_data,

            sample_rate=24000,

            channels=1,

            sample_width=2

        )


        if not wav_data:

            print(
                "❌ WAV conversion failed."
            )

            return None


        print(
            f"✅ WAV audio created: "
            f"{len(wav_data):,} bytes"
        )


        return wav_data


    except Exception as e:

        print(
            "Gemini TTS error:",
            repr(e)
        )


        return None


# ============================================================
# RAG + GEMINI
# ============================================================

def ask_ai(
    question
):

    # ========================================================
    # RAG
    # ========================================================

    retrieved = retrieve_documents(
        question,
        top_k=5
    )


    if retrieved:

        context_parts = []


        for item in retrieved:

            context_parts.append(

                "SOURCE: "
                + str(
                    item["source"]
                )
                + "\n"
                + str(
                    item["text"]
                )

            )


        context = "\n\n".join(
            context_parts
        )


    else:

        context = (
            "No relevant RAG information "
            "was retrieved."
        )


    # ========================================================
    # MEMORY
    # ========================================================

    memory = st.session_state.get(
        "conversation_memory",
        []
    )


    history_parts = []


    for item in memory[-6:]:

        history_parts.append(

            "User: "
            + str(
                item["user"]
            )
            + "\n"
            "Assistant: "
            + str(
                item["assistant"]
            )

        )


    history = "\n\n".join(
        history_parts
    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are the AI assistant inside a
professional AQI Intelligence Dashboard.

Your job is to explain Chennai and
India air-quality information in a
simple, human-friendly way.

IMPORTANT RULES:

1. Use the RAG information as the
   primary factual source.

2. Do not invent AQI values.

3. Do not invent pollutant values.

4. Answer the user's actual question.

5. If the user asks about Chennai,
   focus on Chennai.

6. If the user asks about India,
   focus on India.

7. If the user asks for a comparison,
   compare clearly.

8. Explain what the AQI means.

9. Mention the AQI category when useful.

10. Use simple language.

11. Do not repeat the question.

12. Do not say you are an AI unless asked.

13. Do not invent missing information.

14. Keep the answer concise but useful.

RAG INFORMATION
================

{context}

CONVERSATION MEMORY
===================

{history}

USER QUESTION
=============

{question}

Now provide a clear human-friendly
AQI explanation.
"""


    # ========================================================
    # GEMINI
    # ========================================================

    answer = call_gemini(
        prompt
    )


    # ========================================================
    # MEMORY
    # ========================================================

    memory.append({

        "user":
            question,

        "assistant":
            answer

    })


    st.session_state.conversation_memory = (
        memory[-10:]
    )


    return answer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">
        🌍 AQI Intelligence Dashboard
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="dashboard-subtitle">
        Chennai vs India Air Quality Analytics
        • Machine Learning • RAG • Gemini AI • Voice Assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🌍 AQI Intelligence"
    )


    st.markdown("---")


    page = st.radio(

        "Navigation",

        [

            "📊 Overview",

            "📈 AQI Analytics",

            "🧪 Pollutants",

            "🏙️ City Comparison",

            "🤖 AI Assistant"

        ]

    )


    st.markdown("---")


    if client:

        st.success(
            "🟢 Gemini Connected"
        )

    else:

        st.error(
            "🔴 Gemini API Key Missing"
        )


    if rag_documents:

        st.success(
            "🧠 RAG Ready: "
            + f"{len(rag_documents):,} documents"
        )

    else:

        st.warning(
            "🧠 RAG unavailable"
        )


    st.markdown("---")

    st.caption(
        "AI-powered AQI analytics"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.subheader(
        "📊 Air Quality Overview"
    )


    if df.empty:

        st.error(
            "Dashboard data not found."
        )


    else:

        aqi_col = get_aqi_column(
            df
        )


        if aqi_col:

            values = pd.to_numeric(
                df[aqi_col],
                errors="coerce"
            ).dropna()


            if len(values):

                current = float(
                    values.iloc[-1]
                )

                average = float(
                    values.mean()
                )

                highest = float(
                    values.max()
                )

                lowest = float(
                    values.min()
                )

            else:

                current = 0
                average = 0
                highest = 0
                lowest = 0

        else:

            current = 0
            average = 0
            highest = 0
            lowest = 0


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Current AQI",
                f"{current:.1f}"
            )

            st.caption(
                aqi_category(
                    current
                )
            )


        with c2:

            st.metric(
                "Average AQI",
                f"{average:.1f}"
            )


        with c3:

            st.metric(
                "Highest AQI",
                f"{highest:.1f}"
            )


        with c4:

            st.metric(
                "Total Records",
                f"{len(df):,}"
            )


        st.markdown("")


        if aqi_col:

            st.subheader(
                "📈 AQI Trend"
            )


            chart = df[
                [aqi_col]
            ].copy()


            chart[aqi_col] = pd.to_numeric(
                chart[aqi_col],
                errors="coerce"
            )


            chart = chart.dropna()


            st.line_chart(
                chart.tail(300)
            )


# ============================================================
# AQI ANALYTICS
# ============================================================

elif page == "📈 AQI Analytics":

    st.subheader(
        "📈 AQI Analytics"
    )


    if df.empty:

        st.warning(
            "No dashboard data available."
        )

    else:

        aqi_col = get_aqi_column(
            df
        )


        if aqi_col:

            chart = df[
                [aqi_col]
            ].copy()


            chart[aqi_col] = pd.to_numeric(
                chart[aqi_col],
                errors="coerce"
            )


            chart = chart.dropna()


            st.line_chart(
                chart.tail(500)
            )


            st.subheader(
                "AQI Statistics"
            )


            statistics = pd.DataFrame({

                "Metric": [

                    "Minimum AQI",

                    "Average AQI",

                    "Maximum AQI",

                    "Median AQI"

                ],

                "Value": [

                    chart[aqi_col].min(),

                    chart[aqi_col].mean(),

                    chart[aqi_col].max(),

                    chart[aqi_col].median()

                ]

            })


            st.dataframe(
                statistics,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.warning(
                "AQI column not found."
            )


# ============================================================
# POLLUTANTS
# ============================================================

elif page == "🧪 Pollutants":

    st.subheader(
        "🧪 Pollutant Analysis"
    )


    pollutants = [

        "pm2_5",

        "pm10",

        "no2",

        "so2",

        "co",

        "o3"

    ]


    available = [

        col

        for col in pollutants

        if col in df.columns

    ]


    if available:

        statistics = (

            df[available]

            .apply(
                pd.to_numeric,
                errors="coerce"
            )

            .describe()

            .T

        )


        st.dataframe(
            statistics,
            use_container_width=True
        )


        for pollutant in available:

            st.write(
                f"### {pollutant.upper()}"
            )


            values = pd.to_numeric(
                df[pollutant],
                errors="coerce"
            ).dropna()


            st.line_chart(
                values.tail(300)
            )


    else:

        st.warning(
            "Pollutant columns were not found."
        )


# ============================================================
# CITY COMPARISON
# ============================================================

elif page == "🏙️ City Comparison":

    st.subheader(
        "🏙️ City Comparison"
    )


    city_col = next(

        (

            col

            for col in df.columns

            if col in [

                "city",

                "location",

                "station"

            ]

        ),

        None

    )


    aqi_col = get_aqi_column(
        df
    )


    if (
        city_col
        and aqi_col
    ):

        comparison = (

            df.groupby(
                city_col
            )[aqi_col]

            .mean()

            .sort_values(
                ascending=False
            )

        )


        st.bar_chart(
            comparison
        )


        comparison_table = pd.DataFrame({

            "City":
                comparison.index,

            "Average AQI":
                comparison.values

        })


        comparison_table[
            "Category"
        ] = (

            comparison_table[
                "Average AQI"
            ]

            .apply(
                aqi_category
            )

        )


        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "City information is unavailable."
        )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "🤖 AI Assistant":

    st.subheader(
        "🤖 AQI AI Assistant"
    )


    st.write(
        "Ask questions about Chennai and India "
        "air quality using text or your voice."
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # TEXT CHAT
    # ========================================================

    question = st.chat_input(
        "Ask an AQI question..."
    )


    if question:

        st.session_state.messages.append({

            "role":
                "user",

            "content":
                question

        })


        with st.spinner(
            "🧠 RAG + Gemini are analyzing the AQI data..."
        ):

            answer = ask_ai(
                question
            )


        st.session_state.messages.append({

            "role":
                "assistant",

            "content":
                answer

        })


        st.rerun()


    # ========================================================
    # VOICE ASSISTANT
    # ========================================================

    st.markdown("---")


    st.subheader(
        "🎙️ Voice AQI Assistant"
    )


    st.markdown(
        """
        <div class="voice-card">

        <b>🎙️ Voice Intelligence Pipeline</b>

        <br><br>

        🎙️ Speak your AQI question

        <br>↓

        📝 Gemini understands your speech

        <br>↓

        🧠 RAG retrieves AQI information

        <br>↓

        🤖 Gemini generates the explanation

        <br>↓

        🔊 Gemini converts explanation to speech

        <br>↓

        ▶️ Hear Gemini's answer

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("")


    # ========================================================
    # AUDIO INPUT
    # ========================================================

    audio = st.audio_input(
        "🎙️ Record your AQI question"
    )


    if audio is not None:

        audio_bytes = audio.getvalue()


        if len(audio_bytes) == 0:

            st.error(
                "❌ Empty recording detected."
            )

        else:

            st.caption(
                f"🎙️ Recording received: "
                f"{len(audio_bytes):,} bytes"
            )


            # =================================================
            # BUTTON
            # =================================================

            if st.button(
                "🤖 Understand & Answer",
                type="primary"
            ):

                # =============================================
                # STEP 1 — SPEECH TO TEXT
                # =============================================

                with st.spinner(
                    "🎙️ Gemini is understanding your voice..."
                ):

                    voice_question = (
                        transcribe_voice(
                            audio_bytes
                        )
                    )


                if voice_question:

                    st.success(
                        "🗣️ Voice question understood"
                    )


                    st.info(
                        voice_question
                    )


                    # =========================================
                    # SAVE USER MESSAGE
                    # =========================================

                    st.session_state.messages.append({

                        "role":
                            "user",

                        "content":
                            "🎙️ " + voice_question

                    })


                    # =========================================
                    # STEP 2 — RAG + GEMINI
                    # =========================================

                    with st.spinner(
                        "🧠 RAG + Gemini are generating the AQI explanation..."
                    ):

                        answer = ask_ai(
                            voice_question
                        )


                    # =========================================
                    # TEXT ANSWER
                    # =========================================

                    st.markdown("---")


                    st.markdown(
                        "### 🤖 AQI Explanation"
                    )


                    st.markdown(
                        answer
                    )


                    # =========================================
                    # SAVE ANSWER
                    # =========================================

                    st.session_state.messages.append({

                        "role":
                            "assistant",

                        "content":
                            answer

                    })


                    # =========================================
                    # STEP 3 — GEMINI TTS
                    # =========================================

                    with st.spinner(
                        "🔊 Gemini is generating the spoken explanation..."
                    ):

                        speech = (
                            generate_gemini_speech(
                                answer
                            )
                        )


                    # =========================================
                    # STEP 4 — PLAY GEMINI ANSWER
                    # =========================================

                    if speech:

                        st.markdown("---")


                        st.markdown(
                            "### 🔊 Gemini Voice Answer"
                        )


                        st.success(
                            "✅ Gemini generated the spoken AQI explanation."
                        )


                        # -------------------------------------
                        # IMPORTANT
                        #
                        # ONLY Gemini-generated WAV is played.
                        #
                        # The original user recording is NEVER
                        # passed to st.audio().
                        # -------------------------------------

                        st.audio(
                            speech,
                            format="audio/wav"
                        )


                        st.caption(
                            "▶️ Press play to hear Gemini's explanation."
                        )


                    else:

                        st.warning(
                            "⚠️ Text explanation was generated, "
                            "but Gemini TTS did not return audio."
                        )


                        st.info(
                            "The dashboard will still provide "
                            "the written AQI explanation."
                        )


                else:

                    st.error(
                        "❌ Gemini could not understand the recording."
                    )


                    st.info(
                        "Please record again and speak clearly."
                    )


    # ========================================================
    # CLEAR CONVERSATION
    # ========================================================

    st.markdown("---")


    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.messages = []

        st.session_state.conversation_memory = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "🌍 AQI Intelligence Dashboard | "
    "Machine Learning + RAG + Gemini AI + Voice Assistant"
)

