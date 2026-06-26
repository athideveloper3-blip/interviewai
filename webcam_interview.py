import os
import re
import json
import time
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st

from webcam_component import render_webcam_monitor, get_final_bl_summary, reset_analyzer, announce_question, announce_summary

from streamlit_mic_recorder import mic_recorder
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Pre-load analyzer so webcam starts faster
from webcam_component import get_analyzer
get_analyzer()  # loads MediaPipe

# Pre-load YOLO model at startup
try:
    from ultralytics import YOLO
    import os
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    _yolo_preload = YOLO(model_path)
except Exception:
    pass

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InterviewAI — Resume-Aware Voice Practice",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.error("groq package not found. Run: pip install groq")
    st.stop()

try:
    import pdfplumber
    PDF_READ_AVAILABLE = True
except ImportError:
    PDF_READ_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_WRITE_AVAILABLE = True
except ImportError:
    PDF_WRITE_AVAILABLE = False

# ── Groq Client ───────────────────────────────────────────────────────────────
MODEL        = "llama-3.3-70b-versatile"
WHISPER      = "whisper-large-v3"
VISION_MODEL = "llama-3.2-11b-vision-preview"

@st.cache_resource
def get_client(api_key: str):
    return Groq(api_key=api_key)

def resolve_api_key() -> str:
    """Priority: session UI input → .env / Streamlit secrets → empty."""
    if st.session_state.get("groq_api_key", "").strip():
        return st.session_state["groq_api_key"].strip()
    return os.environ.get("GROQ_API_KEY", "").strip()

def get_groq_client():
    key = resolve_api_key()
    if not key:
        return None
    return get_client(key)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Same dark editorial aesthetic + new resume/webcam components
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --ink:      #06060c;
  --ink2:     #0e0e18;
  --ink3:     #181825;
  --ink4:     #21212f;
  --cream:    #f0ece4;
  --cream2:   rgba(240,236,228,.7);
  --cream3:   rgba(240,236,228,.35);
  --cream4:   rgba(240,236,228,.16);
  --lime:     #c6ff4e;
  --lime-dim: rgba(198,255,78,.18);
  --lime-glow:rgba(198,255,78,.35);
  --red:      #ff5a5a;
  --amber:    #ffb740;
  --blue:     #5ab0ff;
  --purple:   #b794f4;
  --border:   rgba(240,236,228,.07);
  --border2:  rgba(240,236,228,.14);
  --border3:  rgba(240,236,228,.26);
  --serif:    'Playfair Display', Georgia, serif;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'IBM Plex Sans', sans-serif;
  --rad:      12px;
  --shadow:   0 4px 32px rgba(0,0,0,.6);
}

*, *::before, *::after { box-sizing: border-box; }

.stApp, .stApp > div {
  background: var(--ink) !important;
  font-family: var(--sans) !important;
  color: var(--cream) !important;
}

.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(198,255,78,.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(198,255,78,.015) 1px, transparent 1px);
  background-size: 64px 64px;
}

.stApp::after {
  content: '';
  position: fixed; top: -300px; right: -200px; z-index: 0; pointer-events: none;
  width: 700px; height: 700px; border-radius: 50%;
  background: radial-gradient(circle, rgba(198,255,78,.055) 0%, transparent 68%);
}

.block-container {
  padding: 2.5rem 2.5rem 5rem !important;
  max-width: 960px !important;
  position: relative; z-index: 1;
}

h1, h2, h3, h4 { font-family: var(--serif) !important; letter-spacing: -.01em; }

section[data-testid="stSidebar"] {
  background: var(--ink2) !important;
  border-right: 1px solid var(--border) !important;
}

.stButton > button {
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  border-radius: var(--rad) !important;
  transition: all .22s ease !important;
  border: none !important;
  letter-spacing: .01em !important;
}
.stButton > button[kind="primary"] {
  background: var(--lime) !important;
  color: var(--ink) !important;
  box-shadow: 0 0 48px rgba(198,255,78,.3) !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 72px rgba(198,255,78,.55) !important;
  transform: translateY(-2px) !important;
}
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--cream2) !important;
  border: 1px solid var(--border2) !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--border3) !important;
  color: var(--cream) !important;
  background: rgba(240,236,228,.04) !important;
}
.stButton > button:disabled { opacity: .3 !important; }

.stSelectbox > div > div,
.stTextArea > div > div,
.stTextInput > div > div {
  background: var(--ink2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--rad) !important;
  color: var(--cream) !important;
  font-family: var(--sans) !important;
}
.stSelectbox > div > div:focus-within,
.stTextArea > div > div:focus-within,
.stTextInput > div > div:focus-within {
  border-color: rgba(198,255,78,.45) !important;
  box-shadow: 0 0 0 3px rgba(198,255,78,.08) !important;
}
.stSelectbox label, .stTextArea label, .stTextInput label {
  color: var(--cream3) !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
}

div[data-testid="metric-container"] {
  background: var(--ink2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--rad) !important;
  padding: 22px 18px !important;
  text-align: center !important;
  transition: transform .2s, box-shadow .2s !important;
}
div[data-testid="metric-container"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow) !important;
}
div[data-testid="metric-container"] label {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  color: var(--cream4) !important;
  letter-spacing: .15em !important;
  text-transform: uppercase !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  font-family: var(--serif) !important;
  font-size: 44px !important;
  color: var(--lime) !important;
  line-height: 1.1 !important;
}

.stProgress > div > div > div {
  background: var(--lime) !important;
  box-shadow: 0 0 12px rgba(198,255,78,.6) !important;
}
.stProgress > div > div { background: var(--border) !important; }

[data-testid="stFileUploader"] {
  background: var(--ink2) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: var(--rad) !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(198,255,78,.3) !important; }
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
  color: var(--cream3) !important;
  font-family: var(--sans) !important;
}

div[data-testid="stAlert"] {
  background: rgba(255,183,64,.07) !important;
  border: 1px solid rgba(255,183,64,.25) !important;
  border-radius: var(--rad) !important;
}
div[data-testid="stAlert"] p { color: var(--cream2) !important; }

/* ── Custom Cards ── */
.card-hero {
  background: linear-gradient(135deg, var(--ink2) 0%, var(--ink3) 100%);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 40px 44px;
  margin-bottom: 40px;
  position: relative;
  overflow: hidden;
}
.card-hero::after {
  content: '';
  position: absolute; top: -50%; right: -10%;
  width: 380px; height: 380px; border-radius: 50%;
  background: radial-gradient(circle, rgba(198,255,78,.06) 0%, transparent 65%);
  pointer-events: none;
}

.resume-card {
  background: var(--ink2);
  border: 1px solid rgba(183,148,244,.25);
  border-left: 4px solid var(--purple);
  border-radius: var(--rad);
  padding: 22px 28px;
  margin: 14px 0;
}
.resume-card-kicker {
  font-family: var(--mono); font-size: 10px; color: var(--purple);
  letter-spacing: .18em; text-transform: uppercase; margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.resume-card-kicker::before {
  content: ''; display: inline-block; width: 6px; height: 6px;
  background: var(--purple); border-radius: 50%;
}
.resume-snippet {
  font-size: 13px; color: var(--cream3); line-height: 1.85;
  max-height: 120px; overflow: hidden;
  -webkit-mask-image: linear-gradient(to bottom, black 60%, transparent);
}

.q-card {
  background: var(--ink2);
  border: 1px solid var(--border2);
  border-left: 4px solid var(--lime);
  border-radius: var(--rad);
  padding: 26px 30px;
  margin: 14px 0;
  font-size: 19px;
  line-height: 1.75;
  color: var(--cream);
  font-family: var(--serif);
  font-style: italic;
  box-shadow: var(--shadow);
  position: relative;
}
.q-card::before {
  content: '"';
  position: absolute; top: 16px; right: 22px;
  font-family: var(--serif); font-size: 72px; line-height: 1;
  color: rgba(198,255,78,.07);
  pointer-events: none;
}
.q-kicker {
  font-family: var(--mono); font-size: 10px; color: var(--lime);
  letter-spacing: .18em; text-transform: uppercase;
  margin-bottom: 12px; font-style: normal;
  display: flex; align-items: center; gap: 8px;
}
.q-kicker::before {
  content: ''; display: inline-block; width: 6px; height: 6px;
  background: var(--lime); border-radius: 50%;
  animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity: 1; transform: scale(1); }
  50%      { opacity: .4; transform: scale(.7); }
}

.q-source-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 10px;
  padding: 4px 12px; border-radius: 20px;
  margin-bottom: 12px; font-style: normal;
}
.q-source-resume {
  background: rgba(183,148,244,.1); color: var(--purple);
  border: 1px solid rgba(183,148,244,.25);
}
.q-source-general {
  background: var(--lime-dim); color: rgba(198,255,78,.8);
  border: 1px solid rgba(198,255,78,.2);
}

.fb-card {
  background: var(--ink3); border: 1px solid var(--border);
  border-radius: var(--rad); padding: 24px 28px;
  margin: 10px 0; animation: fade-up .35s ease both;
}
@keyframes fade-up {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fb-kicker { font-family: var(--mono); font-size: 10px; color: var(--cream4); letter-spacing: .16em; text-transform: uppercase; margin-bottom: 10px; }
.fb-text { font-size: 14px; color: var(--cream2); line-height: 1.88; }
.fb-tip { margin-top: 14px; padding: 12px 16px; background: var(--lime-dim); border: 1px solid var(--lime-glow); border-radius: 8px; font-size: 13px; color: rgba(198,255,78,.9); line-height: 1.65; }
.kw-hit { display: inline-block; background: rgba(198,255,78,.09); color: rgba(198,255,78,.88); border: 1px solid rgba(198,255,78,.22); font-family: var(--mono); font-size: 11px; padding: 3px 10px; border-radius: 20px; margin: 4px 3px 0; }
.kw-miss { display: inline-block; background: rgba(255,90,90,.07); color: rgba(255,120,120,.82); border: 1px solid rgba(255,90,90,.22); font-family: var(--mono); font-size: 11px; padding: 3px 10px; border-radius: 20px; margin: 4px 3px 0; }

/* Body Language Card */
.bl-card {
  background: var(--ink3);
  border: 1px solid rgba(90,176,255,.2);
  border-left: 4px solid var(--blue);
  border-radius: var(--rad);
  padding: 22px 28px; margin: 14px 0;
  animation: fade-up .35s ease both;
}
.bl-kicker { font-family: var(--mono); font-size: 10px; color: var(--blue); letter-spacing: .18em; text-transform: uppercase; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
.bl-kicker::before { content: ''; display: inline-block; width: 6px; height: 6px; background: var(--blue); border-radius: 50%; }
.bl-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.bl-pill {
  display: flex; align-items: center; gap: 8px; padding: 8px 14px;
  border-radius: 20px; font-family: var(--mono); font-size: 12px;
  border: 1px solid rgba(90,176,255,.2); background: rgba(90,176,255,.05);
  color: var(--cream2);
}
.bl-pill-good { border-color: rgba(198,255,78,.25); background: rgba(198,255,78,.07); color: rgba(198,255,78,.85); }
.bl-pill-warn { border-color: rgba(255,183,64,.25); background: rgba(255,183,64,.07); color: rgba(255,183,64,.85); }
.bl-pill-bad  { border-color: rgba(255,90,90,.25);  background: rgba(255,90,90,.07);  color: rgba(255,90,90,.85); }

.hist-item { display: flex; gap: 16px; align-items: flex-start; background: var(--ink2); border: 1px solid var(--border); border-radius: var(--rad); padding: 16px 22px; margin-bottom: 10px; transition: border-color .2s, transform .2s; }
.hist-item:hover { border-color: var(--border2); transform: translateX(4px); }
.hist-score-good { font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--lime); min-width: 54px; }
.hist-score-mid  { font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--amber); min-width: 54px; }
.hist-score-low  { font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--red); min-width: 54px; }
.hist-q  { font-size: 14px; color: var(--cream); font-weight: 500; margin-bottom: 5px; line-height: 1.5; }
.hist-fb { font-size: 13px; color: var(--cream3); line-height: 1.65; }

.nav-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 52px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
.nav-logo { font-family: var(--mono); font-size: 14px; color: var(--lime); letter-spacing: .18em; text-transform: uppercase; }
.nav-logo span { color: var(--cream4); }
.nav-pill { font-family: var(--mono); font-size: 11px; color: var(--cream3); border: 1px solid var(--border2); padding: 5px 14px; border-radius: 20px; letter-spacing: .06em; }

.tag { display: inline-block; font-family: var(--mono); font-size: 10px; color: var(--cream3); border: 1px solid var(--border); padding: 4px 12px; border-radius: 20px; letter-spacing: .06em; margin-right: 6px; }

.assess-box { background: var(--ink2); border: 1px solid var(--border); border-left: 4px solid var(--lime); border-radius: var(--rad); padding: 26px 30px; margin: 18px 0 28px; }
.assess-kicker { font-family: var(--mono); font-size: 10px; color: var(--lime); letter-spacing: .18em; text-transform: uppercase; margin-bottom: 14px; }
.assess-text { font-size: 15px; color: var(--cream2); line-height: 1.92; }

.section-label { font-family: var(--mono); font-size: 10px; color: var(--cream4); letter-spacing: .18em; text-transform: uppercase; margin: 20px 0 10px; display: flex; align-items: center; gap: 12px; }
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }
a { color: var(--lime) !important; }
div[data-testid="stSpinner"] > div { border-top-color: var(--lime) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "screen":             "setup",
        "groq_api_key":       "",
        "cfg":                {},
        "resume_text":        "",
        "resume_summary":     "",
        "questions":          [],
        "answers":            [],
        "feedbacks":          [],
        "body_lang_results":  [],
        "last_audio_id":      -1,
        "current_q":          0,
        "current_question":   "",
        "current_q_source":   "general",
        "current_feedback":   None,
        "current_bl":         None,
        "summary_text":       "",
        "voice_answer":       "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# PDF Resume Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_resume_text(pdf_bytes: bytes) -> str:
    """Extract text from uploaded PDF resume using pdfplumber."""
    if not PDF_READ_AVAILABLE:
        return ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def summarize_resume(resume_text: str) -> dict:
    """Parse resume to extract structured info for question generation."""
    prompt = f"""You are parsing a candidate's resume to extract key information for an interview.

Resume text:
{resume_text[:4000]}

Extract and return ONLY valid JSON (no markdown, no preamble):
{{
  "name": "candidate name or 'Candidate'",
  "current_role": "most recent job title",
  "years_experience": "estimated years",
  "tech_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "notable_projects": [
    {{"name": "project name", "description": "1 sentence description", "tech": ["tech1","tech2"]}}
  ],
  "companies": ["company1", "company2"],
  "education": "highest degree and field",
  "key_achievements": ["achievement1", "achievement2"],
  "summary": "2 sentence professional summary"
}}

Extract up to 3 notable projects, top 8 tech skills. Be accurate to the resume."""
    raw = ask_llm(prompt, 800)
    raw = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        raw = m.group()
    try:
        return json.loads(raw)
    except Exception:
        return {"name": "Candidate", "summary": resume_text[:200], "tech_skills": [], "notable_projects": [], "companies": []}


# ══════════════════════════════════════════════════════════════════════════════
# Groq AI Helpers
# ══════════════════════════════════════════════════════════════════════════════
def ask_llm(prompt: str, max_tokens: int = 800) -> str:
    c = get_groq_client()
    res = c.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content.strip()


def transcribe_audio(audio_bytes: bytes, filename: str = "answer.wav") -> str:
    try:
        c = get_groq_client()
        transcription = c.audio.transcriptions.create(
            file=(filename, audio_bytes, "audio/wav"),
            model=WHISPER,
            language="en",
            response_format="text",
            prompt="This is an interview answer about software engineering, coding, system design, or behavioral topics.",
        )
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def gen_question(cfg: dict, previous_questions: list, question_number: int, resume_summary: dict) -> tuple[str, str]:
    """
    Returns (question_text, source) where source is 'resume' or 'general'.
    Alternates resume-based and general questions if resume is available.
    """
    has_resume = bool(resume_summary and resume_summary.get("tech_skills"))
    total = cfg["total_questions"]
    
    # Strategy: ~60% resume-based if resume available
    use_resume = has_resume and (question_number % 3 != 0)

    past = ""
    if previous_questions:
        past = "Previously asked — do NOT repeat:\n" + \
               "\n".join(f"  {i+1}. {q}" for i, q in enumerate(previous_questions)) + "\n\n"

    if use_resume:
        projects_str = ""
        if resume_summary.get("notable_projects"):
            projects_str = "Their projects: " + "; ".join(
                f"{p['name']} ({', '.join(p.get('tech', []))})" 
                for p in resume_summary["notable_projects"][:3]
            )
        skills_str = ", ".join(resume_summary.get("tech_skills", [])[:6])
        
        prompt = f"""You are a senior technical interviewer. You have READ this candidate's resume.

Candidate: {resume_summary.get('current_role', 'Software Engineer')} with {resume_summary.get('years_experience', 'several')} years experience.
Skills: {skills_str}
{projects_str}
Companies: {', '.join(resume_summary.get('companies', [])[:3])}
Achievement: {resume_summary.get('key_achievements', ['N/A'])[0] if resume_summary.get('key_achievements') else 'N/A'}

{past}Generate question {question_number} of {total} that DIRECTLY references something specific from their resume.

Examples of good resume-based questions:
- "I see you built [project] — what was the biggest scaling challenge you faced?"
- "You worked at [company] — how did you handle deployments there?"
- "Your resume mentions [skill] — can you walk me through a time you used it under pressure?"

RULES:
- Max 25 words, 1 sentence
- MUST reference something real from their resume (project, company, skill, or achievement)
- Sound conversational, like a real interviewer who read their resume
- {cfg['difficulty']} level depth

Return ONLY the question text."""
        source = "resume"
    else:
        prompt = f"""You are a senior technical interviewer at a {cfg['company_type']} company.
Conducting a {cfg['difficulty']} {cfg['interview_type']} interview for {cfg['role']}.
Focus area: {cfg['focus_area']}.

{past}Generate question {question_number} of {total}.

RULES:
- Max 20 words, 1 sentence only
- No sub-parts, no bullet points
- Conversational, like a real interviewer speaking
- Appropriate for {cfg['difficulty']} level

Return ONLY the question text."""
        source = "general"

    question = ask_llm(prompt, 300)
    return question, source


def gen_feedback(cfg: dict, question: str, answer: str, question_number: int, resume_summary: dict) -> dict:
    resume_context = ""
    if resume_summary and resume_summary.get("tech_skills"):
        resume_context = f"\nCandidate background: {resume_summary.get('summary', '')} Skills: {', '.join(resume_summary.get('tech_skills', [])[:5])}"

    prompt = f"""You are an experienced {cfg['role']} interviewer at a {cfg['company_type']} company.
Interview level: {cfg['difficulty']}. Type: {cfg['interview_type']}. Focus: {cfg['focus_area']}.{resume_context}

Question asked: {question}
Candidate's answer: {answer}

Evaluate and return ONLY valid JSON (no markdown fences):
{{
  "feedback": "3-4 sentence evaluation: what was strong, what was weak or missing",
  "tip": "one specific, actionable improvement",
  "clarity": <integer 1-10>,
  "depth": <integer 1-10>,
  "overall": <integer 1-10>,
  "keywords_mentioned": ["concept1", "concept2"],
  "keywords_missing": ["concept3", "concept4"]
}}

Scoring for {cfg['difficulty']}: Weak: 3-5 | Adequate: 6-7 | Strong: 8-10. Be honest."""
    raw = ask_llm(prompt, 700)
    raw = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        raw = m.group()
    return json.loads(raw)


def analyze_body_language(frame_b64: str, question: str) -> dict:
    """
    Send webcam frame to Groq vision model for body language analysis.
    Returns structured body language feedback.
    """
    prompt = f"""You are analyzing a job interview candidate's body language from a single webcam frame.

The candidate just answered this interview question: "{question}"

Analyze the image carefully and return ONLY valid JSON (no markdown):
{{
  "eye_contact": <integer 1-10>,
  "confidence": <integer 1-10>,
  "posture": <integer 1-10>,
  "expression": "brief description of facial expression (e.g. 'calm and focused', 'slightly nervous', 'confident smile')",
  "overall_presence": <integer 1-10>,
  "observations": ["observation1", "observation2", "observation3"],
  "tip": "one specific body language improvement for interviews"
}}

Be honest and constructive. Base scores only on what you can actually see.
If image quality is poor or face is not clearly visible, give neutral scores (5-6) and note this."""

    try:
        c = get_groq_client()
        res = c.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{frame_b64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            raw = m.group()
        return json.loads(raw)
    except Exception as e:
        return {
            "eye_contact": 5, "confidence": 5, "posture": 5,
            "expression": "Could not analyze",
            "overall_presence": 5,
            "observations": [f"Analysis unavailable: {str(e)[:80]}"],
            "tip": "Maintain eye contact with the camera and sit up straight."
        }


def gen_summary(cfg: dict, questions: list, answers: list, feedbacks: list,
                body_lang_results: list, resume_summary: dict) -> str:
    scored = [f for f in feedbacks if f]
    if not scored:
        return "No answers were submitted during this session."

    ac = round(sum(f["clarity"] for f in scored) / len(scored), 1)
    ad = round(sum(f["depth"]   for f in scored) / len(scored), 1)
    ao = round(sum(f["overall"] for f in scored) / len(scored), 1)

    bl_scored = [b for b in body_lang_results if b and b.get("overall_presence")]
    bl_avg = round(sum(b["overall_presence"] for b in bl_scored) / len(bl_scored), 1) if bl_scored else None

    qa_block = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        f = feedbacks[i]
        if f:
            qa_block += f"Q{i+1}: {q}\nScore {f['overall']}/10 — {f['feedback']}\n\n"
        else:
            qa_block += f"Q{i+1}: {q}\n(Skipped)\n\n"

    resume_ctx = ""
    if resume_summary and resume_summary.get("current_role"):
        resume_ctx = f"\nCandidate profile: {resume_summary.get('current_role')} | {resume_summary.get('years_experience', '?')} yrs | Skills: {', '.join(resume_summary.get('tech_skills', [])[:5])}"

    bl_note = f"\nBody language avg presence score: {bl_avg}/10" if bl_avg else ""

    prompt = f"""Summarize this completed {cfg['role']} {cfg['interview_type']} interview ({cfg['difficulty']} level).{resume_ctx}{bl_note}

Average scores — Clarity: {ac}/10  Depth: {ad}/10  Overall: {ao}/10

Q&A breakdown:
{qa_block}
Write a 5-sentence assessment:
1. Overall hiring verdict (lean hire / borderline / no hire) with brief rationale
2. The candidate's clearest strength demonstrated
3. The most critical gap or weakness to address
4. Body language and presence note (if webcam data available)
5. One concrete next step + motivating closing sentence

Be direct, specific, honest. No generic platitudes."""
    return ask_llm(prompt, 700)

def gen_summary(cfg: dict, questions: list, answers: list, feedbacks: list,
                body_lang_results: list, resume_summary: dict) -> str:
    scored = [f for f in feedbacks if f]
    if not scored:
        return "No answers were submitted during this session."

    ac = round(sum(f["clarity"] for f in scored) / len(scored), 1)
    ad = round(sum(f["depth"]   for f in scored) / len(scored), 1)
    ao = round(sum(f["overall"] for f in scored) / len(scored), 1)

    bl_scored = [b for b in body_lang_results if b and b.get("overall_presence")]
    bl_avg = round(sum(b["overall_presence"] for b in bl_scored) / len(bl_scored), 1) if bl_scored else None

    qa_block = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        f = feedbacks[i]
        if f:
            qa_block += f"Q{i+1}: {q}\nScore {f['overall']}/10 — {f['feedback']}\n\n"
        else:
            qa_block += f"Q{i+1}: {q}\n(Skipped)\n\n"

    resume_ctx = ""
    if resume_summary and resume_summary.get("current_role"):
        resume_ctx = f"\nCandidate profile: {resume_summary.get('current_role')} | {resume_summary.get('years_experience', '?')} yrs | Skills: {', '.join(resume_summary.get('tech_skills', [])[:5])}"

    bl_note = f"\nBody language avg presence score: {bl_avg}/10" if bl_avg else ""

    prompt = f"""Summarize this completed {cfg['role']} {cfg['interview_type']} interview ({cfg['difficulty']} level).{resume_ctx}{bl_note}

Average scores — Clarity: {ac}/10  Depth: {ad}/10  Overall: {ao}/10

Q&A breakdown:
{qa_block}
Write a 5-sentence assessment:
1. Overall hiring verdict (lean hire / borderline / no hire) with brief rationale
2. The candidate's clearest strength demonstrated
3. The most critical gap or weakness to address
4. Body language and presence note (if webcam data available)
5. One concrete next step + motivating closing sentence

Be direct, specific, honest. No generic platitudes."""
    return ask_llm(prompt, 700)

def render_alert_banner(body_lang_summary: dict | None):
    """
    Reads a body-language summary dict and surfaces prominent coloured
    alert banners. Called both live (per question) and on the summary screen.

    Severity levels:
      RED    — integrity risk  (multiple faces, phone, face absent)
      ORANGE — suspicious      (sustained look-away, head turned, high cheat risk)
    """
    if not body_lang_summary:
        return

    events   = body_lang_summary.get("cheat_events", [])
    cheat_s  = body_lang_summary.get("cheat_score", 0)
    phone_ev = body_lang_summary.get("phone_events", 0)
    multi_ev = body_lang_summary.get("multiple_face_events", 0)
    absent_f = body_lang_summary.get("face_absent_frames", 0)

    alerts = []  # list of (severity, icon, message)

    # Multiple faces
    if multi_ev > 5:
        alerts.append(("red", "🚨",
            f"Another person detected in frame ({multi_ev} frames) — candidate must be alone."))

    # Phone / mobile usage
    if phone_ev >= 2:
        alerts.append(("red", "📱",
            f"Possible phone/mobile usage detected {phone_ev} time(s) — head-down posture with downward gaze."))

    # Face absent from camera
    if absent_f > 45:   # ~3 seconds at 15 fps
        alerts.append(("red", "👁",
            "Candidate face not visible for an extended period — camera may be obscured or candidate left frame."))

    # Recent high-severity events (excluding ones already covered above)
    skip_types = {"PHONE_USAGE", "LONG_ABSENCE", "MULTIPLE_FACES"}
    recent_high = [e for e in events[-10:]
                   if e["severity"] == "high" and e.get("type") not in skip_types]
    for e in recent_high[-2:]:
        ev_type = e.get("type", "")
        if ev_type == "LOOKING_AWAY" and e.get("duration", 0) > 4:
            alerts.append(("orange", "👀", f"Sustained off-screen gaze: {e['description']}"))
        elif ev_type == "HEAD_TURNED":
            alerts.append(("orange", "↔️", f"Head turned significantly: {e['description']}"))

    # General high cheat risk fallback (only if no specific alert already shown)
    if cheat_s >= 6 and not alerts:
        alerts.append(("orange", "⚠️",
            f"Elevated integrity risk score: {cheat_s}/10 — review the monitoring log."))

    if not alerts:
        return

    for severity, icon, message in alerts[:3]:   # cap at 3 banners
        if phone_ev > 0 and severity == "red":
            play_alert_sound()

        if severity == "red":
            bg       = "rgba(255,90,90,.12)"
            border   = "rgba(255,90,90,.45)"
            color    = "#ff6b6b"
            label_bg = "rgba(255,90,90,.25)"
            label    = "ALERT"
        else:
            bg       = "rgba(255,183,64,.10)"
            border   = "rgba(255,183,64,.40)"
            color    = "#ffb740"
            label_bg = "rgba(255,183,64,.22)"
            label    = "WARNING"

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-left:4px solid {color};
                    border-radius:10px;padding:14px 18px;margin:8px 0;
                    display:flex;align-items:flex-start;gap:12px;
                    animation:fade-up .3s ease both;">
          <span style="font-size:20px;line-height:1;flex-shrink:0;">{icon}</span>
          <div style="flex:1;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
              <span style="background:{label_bg};color:{color};font-family:'IBM Plex Mono',monospace;
                           font-size:10px;font-weight:700;letter-spacing:.14em;
                           padding:3px 9px;border-radius:6px;">{label}</span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                           color:rgba(240,236,228,.35);letter-spacing:.06em;">
                BODY LANGUAGE MONITOR
              </span>
            </div>
            <div style="font-size:13px;color:rgba(240,236,228,.85);line-height:1.65;">
              {message}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

def play_alert_sound():
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
    </audio>
    """, height=0)

def detect_tab_switch():
    components.html("""
    <script>
    document.addEventListener("visibilitychange", function() {

        if (document.hidden) {
            alert("WARNING: Tab switching detected!");

            fetch("/tab_switch_detected", {
                method: "POST"
            });
        }

    });

    window.addEventListener("blur", function() {
        console.log("Window lost focus");
    });
    </script>
    """, height=0)

# ══════════════════════════════════════════════════════════════════════════════
# PDF Report Builder (upgraded with resume + body language)
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(cfg, questions, answers, feedbacks, body_lang_results, summary_text, resume_summary) -> BytesIO:
    if not PDF_WRITE_AVAILABLE:
        raise RuntimeError("reportlab not installed")

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)

    INK    = colors.HexColor("#06060c")
    LIME   = colors.HexColor("#c6ff4e")
    GREY   = colors.HexColor("#555566")
    MID    = colors.HexColor("#888899")
    BODY   = colors.HexColor("#2a2a3a")
    LIGHT  = colors.HexColor("#f0ece4")
    PURPLE = colors.HexColor("#b794f4")
    BLUE   = colors.HexColor("#5ab0ff")

    def ps(name, **kw): return ParagraphStyle(name, **kw)

    title_s = ps("T",  fontName="Helvetica-Bold", fontSize=28, textColor=INK, spaceAfter=4, leading=34)
    sub_s   = ps("S",  fontName="Helvetica",      fontSize=13, textColor=GREY, spaceAfter=3)
    meta_s  = ps("M",  fontName="Helvetica",      fontSize=9,  textColor=MID,  spaceAfter=14)
    h2_s    = ps("H2", fontName="Helvetica-Bold", fontSize=14, textColor=INK,  spaceBefore=18, spaceAfter=7)
    h3_s    = ps("H3", fontName="Helvetica-Bold", fontSize=11, textColor=GREY, spaceBefore=10, spaceAfter=5)
    body_s  = ps("B",  fontName="Helvetica",      fontSize=10, textColor=BODY, leading=17, spaceAfter=4)
    label_s = ps("L",  fontName="Helvetica-Bold", fontSize=9,  textColor=GREY, spaceAfter=2, leading=13)
    fb_s    = ps("FB", fontName="Helvetica",      fontSize=10, textColor=colors.HexColor("#3a3a4a"), leading=16, spaceAfter=4)
    tip_s   = ps("TP", fontName="Helvetica-Oblique", fontSize=10, textColor=colors.HexColor("#2d5e0a"), leading=14, spaceAfter=4)
    sum_s   = ps("SM", fontName="Helvetica",      fontSize=11, textColor=BODY, leading=19, spaceAfter=6)
    foot_s  = ps("F",  fontName="Helvetica",      fontSize=8,  textColor=MID,  alignment=TA_CENTER)
    miss_s  = ps("MS", fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#883333"), spaceAfter=4)
    skip_s  = ps("SK", fontName="Helvetica-Oblique", fontSize=10, textColor=MID, spaceAfter=4)
    pur_s   = ps("PU", fontName="Helvetica",      fontSize=10, textColor=colors.HexColor("#6b46c1"), leading=16, spaceAfter=4)
    bl_s    = ps("BL", fontName="Helvetica",      fontSize=10, textColor=colors.HexColor("#1e5c8a"), leading=16, spaceAfter=4)

    story = []

    # Header
    story.append(Paragraph("InterviewAI Report", title_s))
    story.append(Paragraph(f"{cfg['role']}  ·  {cfg['interview_type']}  ·  {cfg['difficulty']}", sub_s))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  ·  "
        f"{cfg['company_type']}  ·  Focus: {cfg['focus_area']}", meta_s))
    story.append(HRFlowable(width="100%", thickness=2, color=LIME, spaceAfter=18))

    # Resume Section
    if resume_summary and resume_summary.get("current_role"):
        story.append(Paragraph("Candidate Profile", h2_s))
        resume_data = [
            ["Name", resume_summary.get("name", "Candidate")],
            ["Current Role", resume_summary.get("current_role", "N/A")],
            ["Experience", resume_summary.get("years_experience", "N/A")],
            ["Education", resume_summary.get("education", "N/A")],
            ["Top Skills", ", ".join(resume_summary.get("tech_skills", [])[:6])],
            ["Companies", ", ".join(resume_summary.get("companies", [])[:3])],
        ]
        rt = Table(resume_data, colWidths=[40*mm, 130*mm])
        rt.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#6b46c1")),
            ("TEXTCOLOR",   (1, 0), (1, -1), BODY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f7f4fe"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), .5, colors.HexColor("#e0d4f7")),
            ("PADDING",     (0, 0), (-1, -1), 8),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(rt)
        story.append(Spacer(1, 14))

    # Scores table
    story.append(HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#d8d4cc"), spaceAfter=8))
    story.append(Paragraph("Performance Scores", h2_s))
    scored = [f for f in feedbacks if f]
    ac = round(sum(f["clarity"] for f in scored) / len(scored), 1) if scored else 0
    ad = round(sum(f["depth"]   for f in scored) / len(scored), 1) if scored else 0
    ao = round(sum(f["overall"] for f in scored) / len(scored), 1) if scored else 0

    bl_scored = [b for b in body_lang_results if b and b.get("overall_presence")]
    bl_avg = round(sum(b["overall_presence"] for b in bl_scored) / len(bl_scored), 1) if bl_scored else None
    bl_ec  = round(sum(b["eye_contact"] for b in bl_scored) / len(bl_scored), 1) if bl_scored else None
    bl_cf  = round(sum(b["confidence"] for b in bl_scored) / len(bl_scored), 1) if bl_scored else None

    score_data = [["Metric", "Score", "Out Of"]]
    score_data += [
        ["Answer Clarity", f"{ac}", "10"],
        ["Answer Depth",   f"{ad}", "10"],
        ["Overall Answer", f"{ao}", "10"],
    ]
    if bl_avg:
        score_data += [
            ["Eye Contact",     f"{bl_ec}", "10"],
            ["Confidence",      f"{bl_cf}", "10"],
            ["Overall Presence",f"{bl_avg}", "10"],
        ]

    tbl = Table(score_data, colWidths=[80*mm, 40*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), INK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), LIGHT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f4ee"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), .5, colors.HexColor("#d8d4cc")),
        ("PADDING",     (0, 0), (-1, -1), 9),
        ("ALIGN",       (1, 0), (2, -1), "CENTER"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # Summary
    story.append(HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#d8d4cc"), spaceAfter=8))
    story.append(Paragraph("AI Assessment", h2_s))
    story.append(Paragraph(summary_text, sum_s))

    # Question breakdown
    story.append(HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#d8d4cc"), spaceAfter=8))
    story.append(Paragraph("Question Breakdown", h2_s))

    for i, (q, a) in enumerate(zip(questions, answers)):
        f  = feedbacks[i]
        bl = body_lang_results[i] if i < len(body_lang_results) else None

        qt = Table([[f"Q{i+1}", q]], colWidths=[14*mm, 146*mm])
        qt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), INK),
            ("TEXTCOLOR",  (0, 0), (0, 0), LIME),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f0ece4")),
            ("TEXTCOLOR",  (1, 0), (1, 0), INK),
            ("PADDING",    (0, 0), (-1, -1), 9),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(qt)

        if a and a not in ("[skipped]", ""):
            story.append(Paragraph("Candidate Answer:", label_s))
            story.append(Paragraph((a[:1400] + "…") if len(a) > 1400 else a, body_s))

        if f:
            sr = Table(
                [[f"Clarity: {f['clarity']}/10", f"Depth: {f['depth']}/10", f"Overall: {f['overall']}/10"]],
                colWidths=[55*mm, 55*mm, 50*mm],
            )
            sr.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edffa8")),
                ("TEXTCOLOR",  (0, 0), (-1, -1), colors.HexColor("#2a5500")),
                ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                ("PADDING",    (0, 0), (-1, -1), 7),
                ("GRID",       (0, 0), (-1, -1), .5, colors.HexColor("#b8e860")),
            ]))
            story.append(sr)
            story.append(Spacer(1, 5))
            story.append(Paragraph("Feedback:", label_s))
            story.append(Paragraph(f["feedback"], fb_s))
            if f.get("tip"):
                story.append(Paragraph(f"Tip: {f['tip']}", tip_s))
            if f.get("keywords_mentioned"):
                story.append(Paragraph("Covered: " + ", ".join(f["keywords_mentioned"]), label_s))
            if f.get("keywords_missing"):
                story.append(Paragraph("Missed: " + ", ".join(f["keywords_missing"]), miss_s))
        else:
            story.append(Paragraph("— Skipped —", skip_s))

        # Body language
        if bl and bl.get("overall_presence"):
            story.append(Paragraph("Body Language:", label_s))
            bl_row = Table(
                [[f"Eye Contact: {bl['eye_contact']}/10", f"Confidence: {bl['confidence']}/10",
                  f"Posture: {bl['posture']}/10", f"Presence: {bl['overall_presence']}/10"]],
                colWidths=[42*mm, 42*mm, 42*mm, 34*mm],
            )
            bl_row.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f4ff")),
                ("TEXTCOLOR",  (0, 0), (-1, -1), colors.HexColor("#1e5c8a")),
                ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                ("PADDING",    (0, 0), (-1, -1), 7),
                ("GRID",       (0, 0), (-1, -1), .5, colors.HexColor("#a0c8e8")),
            ]))
            story.append(bl_row)
            if bl.get("tip"):
                story.append(Paragraph(f"Body Language Tip: {bl['tip']}", bl_s))

        story.append(Spacer(1, 12))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1.5, color=INK, spaceBefore=14, spaceAfter=9))
    story.append(Paragraph(
        f"InterviewAI · Resume-Aware Edition · Groq + LLaMA 3.3 + Whisper + Vision · {datetime.now().year}",
        foot_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# Webcam Component — captures frame as base64 on demand
# ══════════════════════════════════════════════════════════════════════════════
WEBCAM_HTML = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: 'IBM Plex Sans', sans-serif; }

  .wc-wrap {
    background: #0e0e18;
    border: 1px solid rgba(90,176,255,.2);
    border-radius: 14px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .wc-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; color: rgba(90,176,255,.6);
    letter-spacing: .18em; text-transform: uppercase;
  }

  .video-ring {
    position: relative;
    width: 220px; height: 220px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid rgba(90,176,255,.3);
    box-shadow: 0 0 24px rgba(90,176,255,.15);
    margin: 0 auto;
    background: #06060c;
  }
  video {
    width: 100%; height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
  }
  .ring-overlay {
    position: absolute; inset: 0;
    border-radius: 50%;
    background: radial-gradient(circle at center, transparent 60%, rgba(90,176,255,.08));
    pointer-events: none;
  }
  .rec-dot {
    position: absolute; top: 12px; right: 12px;
    width: 10px; height: 10px; border-radius: 50%;
    background: #ff5a5a; display: none;
    animation: rdot 1s infinite;
  }
  @keyframes rdot { 0%,100%{opacity:1} 50%{opacity:.2} }
  .rec-dot.visible { display: block; }

  .controls {
    display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;
  }
  .btn {
    padding: 8px 16px; border-radius: 8px; border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px; font-weight: 600; cursor: pointer;
    transition: all .18s;
  }
  .btn-start { background: rgba(90,176,255,.15); color: #5ab0ff; border: 1px solid rgba(90,176,255,.3); }
  .btn-start:hover { background: rgba(90,176,255,.25); }
  .btn-capture {
    background: #5ab0ff; color: #06060c;
    box-shadow: 0 0 20px rgba(90,176,255,.4);
  }
  .btn-capture:hover { box-shadow: 0 0 32px rgba(90,176,255,.65); transform: translateY(-1px); }
  .btn-capture:disabled { opacity: .3; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-off { background: transparent; color: rgba(240,236,228,.3); border: 1px solid rgba(240,236,228,.08); font-size: 11px; }

  .status {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: rgba(240,236,228,.3); text-align: center; letter-spacing: .04em;
    min-height: 20px;
  }
  .status.on  { color: #5ab0ff; }
  .status.ok  { color: #c6ff4e; }
  .status.err { color: #ff5a5a; }
</style>

<div class="wc-wrap">
  <div class="wc-title">📹 Webcam · Body Language Analysis</div>
  <div class="video-ring">
    <video id="vid" autoplay muted playsinline></video>
    <div class="ring-overlay"></div>
    <div class="rec-dot" id="rec-dot"></div>
  </div>
  <div class="controls">
    <button class="btn btn-start" id="btn-start" onclick="startCam()">Enable Camera</button>
    <button class="btn btn-capture" id="btn-capture" onclick="captureFrame()" disabled>📸 Capture Frame</button>
    <button class="btn btn-off" onclick="stopCam()">Stop</button>
  </div>
  <div class="status" id="status">Camera off — click Enable to start</div>
</div>
<canvas id="canvas" style="display:none"></canvas>

<script>
let stream = null;

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    document.getElementById('vid').srcObject = stream;
    document.getElementById('btn-capture').disabled = false;
    document.getElementById('rec-dot').className = 'rec-dot visible';
    setStatus('Camera active — capture a frame after answering', 'on');
  } catch(e) {
    setStatus('Camera access denied: ' + e.message, 'err');
  }
}

function stopCam() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  document.getElementById('vid').srcObject = null;
  document.getElementById('btn-capture').disabled = true;
  document.getElementById('rec-dot').className = 'rec-dot';
  setStatus('Camera off', '');
}

function captureFrame() {
  const vid = document.getElementById('vid');
  const cv  = document.getElementById('canvas');
  cv.width = vid.videoWidth || 640;
  cv.height = vid.videoHeight || 480;
  const ctx = cv.getContext('2d');
  ctx.save();
  ctx.translate(cv.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(vid, 0, 0);
  ctx.restore();
  const b64 = cv.toDataURL('image/jpeg', 0.75).split(',')[1];
  setStatus('Frame captured! Sending for analysis…', 'ok');
  window.parent.postMessage({ type: 'streamlit:setComponentValue', value: 'FRAME:' + b64 }, '*');
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status' + (cls ? ' ' + cls : '');
}
</script>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def avg(arr): return round(sum(arr) / len(arr), 1) if arr else 0

def score_class(n):
    if n is None: return "hist-score-low"
    return "hist-score-good" if n >= 7 else ("hist-score-mid" if n >= 5 else "hist-score-low")

def render_nav(subtitle: str = ""):
    st.markdown(f"""
    <div class="nav-bar">
      <div class="nav-logo">INTERVIEW<span>//</span>AI</div>
      <div class="nav-pill">{subtitle or "Resume-Aware · Voice + Vision"}</div>
    </div>
    """, unsafe_allow_html=True)
    

def bl_pill_class(score: int) -> str:
    if score >= 7: return "bl-pill bl-pill-good"
    if score >= 5: return "bl-pill bl-pill-warn"
    return "bl-pill bl-pill-bad"


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: SETUP
# ══════════════════════════════════════════════════════════════════════════════
if S.screen == "setup":
    render_nav("v5.0 · Resume + Webcam")

    st.markdown("""
    <div class="card-hero">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:rgba(198,255,78,.6);
                  letter-spacing:.18em;text-transform:uppercase;margin-bottom:18px;">
        AI Mock Interviews · Resume-Aware · Body Language Analysis
      </div>
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(32px,5vw,52px);
                 line-height:1.07;letter-spacing:-.02em;margin-bottom:16px;color:#f0ece4;">
        Questions from<br/><em style="color:#c6ff4e;">your resume.</em>
      </h1>
      <p style="color:rgba(240,236,228,.52);font-size:15px;max-width:520px;line-height:1.9;">
        Upload your CV and get interviewed on <em>exactly</em> what you've built.
        Speak your answers, get scored on content <em>and</em> body language.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:36px;">
      <span style="background:rgba(183,148,244,.1);border:1px solid rgba(183,148,244,.25);color:rgba(183,148,244,.9);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 14px;border-radius:20px;">📄 Resume-Based Questions</span>
      <span style="background:rgba(90,176,255,.1);border:1px solid rgba(90,176,255,.25);color:rgba(90,176,255,.9);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 14px;border-radius:20px;">👁 Body Language AI</span>
      <span style="background:rgba(198,255,78,.08);border:1px solid rgba(198,255,78,.22);color:rgba(198,255,78,.8);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 14px;border-radius:20px;">🎙 Whisper STT</span>
      <span style="background:rgba(198,255,78,.08);border:1px solid rgba(198,255,78,.22);color:rgba(198,255,78,.8);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 14px;border-radius:20px;">📊 Full PDF Report</span>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key ──
    st.markdown('<div class="section-label">Groq API Key</div>', unsafe_allow_html=True)

    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key:
        st.markdown("""
        <div style="background:rgba(198,255,78,.06);border:1px solid rgba(198,255,78,.18);
                    border-radius:10px;padding:12px 18px;display:flex;align-items:center;gap:10px;">
          <span style="color:#c6ff4e;font-size:16px;">✓</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:rgba(198,255,78,.75);">
            API key loaded from environment / Streamlit secrets
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_key, col_key_help = st.columns([3, 2])
        with col_key:
            typed_key = st.text_input(
                "Groq API Key",
                value=S.groq_api_key,
                type="password",
                placeholder="gsk_...",
                label_visibility="collapsed",
                key="groq_key_input",
                help="Your key is stored in session memory only — never sent anywhere except Groq's API.",
            )
            if typed_key != S.groq_api_key:
                S.groq_api_key = typed_key
        with col_key_help:
            st.markdown("""
            <div style="background:rgba(198,255,78,.04);border:1px solid rgba(198,255,78,.12);
                        border-radius:10px;padding:12px 16px;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:rgba(198,255,78,.5);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">
                Get a free key
              </div>
              <div style="font-size:12px;color:rgba(240,236,228,.45);line-height:1.7;">
                <a href="https://console.groq.com/keys" target="_blank"
                   style="color:rgba(198,255,78,.7);">console.groq.com/keys</a><br/>
                Free tier · Fast inference<br/>
                Key stays in your browser session only.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Live validation indicator
        current_key = resolve_api_key()
        if current_key:
            if current_key.startswith("gsk_") and len(current_key) > 20:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
                  <span style="width:8px;height:8px;background:#c6ff4e;border-radius:50%;display:inline-block;
                               box-shadow:0 0 6px rgba(198,255,78,.8);"></span>
                  <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(198,255,78,.7);">
                    Key looks valid — ready to go
                  </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
                  <span style="width:8px;height:8px;background:#ffb740;border-radius:50%;display:inline-block;"></span>
                  <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(255,183,64,.7);">
                    Key format looks off — Groq keys start with gsk_
                  </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
              <span style="width:8px;height:8px;background:rgba(240,236,228,.2);border-radius:50%;display:inline-block;"></span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(240,236,228,.25);">
                No key entered yet
              </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Resume Upload ──
    st.markdown('<div class="section-label">Upload Your Resume (Optional but Recommended)</div>', unsafe_allow_html=True)
    
    col_up, col_info = st.columns([3, 2])
    with col_up:
        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload your resume PDF — questions will be tailored to your actual experience and projects."
        )
    with col_info:
        st.markdown("""
        <div style="background:rgba(183,148,244,.06);border:1px solid rgba(183,148,244,.15);
                    border-radius:10px;padding:16px 20px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:rgba(183,148,244,.7);
                      letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;">Why upload?</div>
          <div style="font-size:13px;color:rgba(240,236,228,.5);line-height:1.75;">
            Without resume: generic role-based questions<br/>
            <strong style="color:rgba(183,148,244,.8);">With resume:</strong> "I see you built [your project] — explain the architecture"
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Process resume immediately on upload
    if resume_file and not S.resume_text:
        with st.spinner("Extracting and parsing your resume…"):
            if not PDF_READ_AVAILABLE:
                st.error("pdfplumber not installed. Run: pip install pdfplumber")
            else:
                raw_text = extract_resume_text(resume_file.read())
                if raw_text and not raw_text.startswith("[PDF extraction error"):
                    S.resume_text = raw_text
                    resume_data = summarize_resume(raw_text)
                    S.resume_summary = resume_data
                    st.rerun()
                else:
                    st.error(f"Could not extract text from PDF: {raw_text}")

    if S.resume_summary and isinstance(S.resume_summary, dict) and S.resume_summary.get("current_role"):
        rs = S.resume_summary
        skills_html = "".join(f'<span style="background:rgba(183,148,244,.08);border:1px solid rgba(183,148,244,.2);color:rgba(183,148,244,.8);font-family:\'IBM Plex Mono\',monospace;font-size:10px;padding:3px 10px;border-radius:20px;margin:3px;">{s}</span>' for s in rs.get("tech_skills", [])[:8])
        projects_html = ""
        for p in rs.get("notable_projects", [])[:3]:
            tech = ", ".join(p.get("tech", [])[:3])
            projects_html += f'<div style="padding:8px 0;border-bottom:1px solid rgba(183,148,244,.1);"><strong style="color:rgba(183,148,244,.9);">{p["name"]}</strong> <span style="font-size:11px;color:rgba(240,236,228,.4);">({tech})</span><br/><span style="font-size:12px;color:rgba(240,236,228,.5);">{p.get("description","")}</span></div>'

        st.markdown(f"""
        <div class="resume-card">
          <div class="resume-card-kicker">Resume Parsed Successfully</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div>
              <div style="font-size:12px;color:rgba(240,236,228,.4);margin-bottom:4px;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;">Role</div>
              <div style="font-size:15px;color:#f0ece4;font-weight:600;">{rs.get("current_role","N/A")}</div>
              <div style="font-size:12px;color:rgba(240,236,228,.4);margin-top:12px;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;">Skills</div>
              <div style="display:flex;flex-wrap:wrap;gap:4px;">{skills_html}</div>
            </div>
            <div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:rgba(240,236,228,.4);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Notable Projects</div>
              {projects_html or '<div style="color:rgba(240,236,228,.3);font-size:12px;">No projects detected</div>'}
            </div>
          </div>
          <div style="margin-top:14px;font-size:12px;color:rgba(183,148,244,.7);font-style:italic;">
            ✓ Interview questions will reference your specific experience
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif S.resume_text == "":
        st.caption("No resume uploaded — questions will be role-based only")

    # ── Interview Config ──
    st.markdown('<div class="section-label">Configure Interview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox("Target Role", [
            "Software Engineer", "Backend Engineer", "Frontend Engineer",
            "Full Stack Engineer", "ML / AI Engineer", "Data Scientist",
            "Data Engineer", "DevOps / SRE", "Cloud Engineer",
            "Product Manager", "Engineering Manager", "Security Engineer",
        ])
    with col2:
        itype = st.selectbox("Interview Type", [
            "Technical", "Behavioral", "System Design",
            "Coding Concepts", "Mixed (Tech + Behavioral)",
        ])
    with col3:
        difficulty = st.selectbox("Level", [
            "Entry Level (0-2 yrs)", "Intermediate (2-5 yrs)",
            "Senior (5-8 yrs)", "Staff / Principal (8+ yrs)",
        ])

    col4, col5, col6 = st.columns(3)
    with col4:
        company = st.selectbox("Company Type", [
            "FAANG / Big Tech", "Series A/B Startup",
            "Unicorn / Late-Stage", "Mid-size Tech", "Consulting / Agency", "Any",
        ])
    with col5:
        focus = st.selectbox("Focus Area", [
            "General", "Algorithms & Data Structures",
            "OOP & Design Patterns", "Databases & SQL",
            "APIs & Microservices", "System Architecture",
            "Leadership & Culture", "Cloud & Infrastructure",
            "Problem Solving", "Communication & Collaboration",
        ])
    with col6:
        qty_option = st.selectbox("Session Length", [
            ("3 questions — Quick (≈15 min)", 3),
            ("5 questions — Standard (≈30 min)", 5),
            ("8 questions — Full (≈50 min)", 8),
            ("10 questions — Extended (≈65 min)", 10),
        ], format_func=lambda x: x[0], index=1)

    st.markdown('<div class="section-label">Features</div>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        use_whisper = st.toggle("Groq Whisper Transcription", value=True)
    with col_f2:
        use_webcam = st.toggle("Webcam Body Language Analysis", value=True)
    with col_f3:
        auto_tts = st.toggle("Auto-play questions aloud", value=True)

    st.write("")
    has_key = bool(resolve_api_key())
    if not has_key:
        st.markdown("""
        <div style="text-align:center;padding:12px;border:1px solid rgba(255,183,64,.2);
                    border-radius:10px;background:rgba(255,183,64,.05);margin-bottom:8px;">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(255,183,64,.7);">
            ⚠ Enter your Groq API key above to start
          </span>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Begin Interview  →", type="primary", use_container_width=True, disabled=not has_key):
        S.cfg = {
            "role": role, "interview_type": itype, "difficulty": difficulty,
            "company_type": company, "focus_area": focus,
            "total_questions": qty_option[1],
            "use_whisper": use_whisper, "use_webcam": use_webcam, "auto_tts": auto_tts,
        }
        S.questions = []; S.answers = []; S.feedbacks = []
        S.body_lang_results = []
        S.current_q = 0; S.current_question = ""
        S.current_q_source = "general"
        S.current_feedback = None; S.current_bl = None
        S.summary_text = ""; S.voice_answer = ""
        S.last_audio_id = -1
        S.screen = "interview"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: INTERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif S.screen == "interview":
    cfg   = S.cfg
    total = cfg["total_questions"]
    cq    = S.current_q
    pct   = int((cq / total) * 100)
    rs    = S.resume_summary if isinstance(S.resume_summary, dict) else {}

    render_nav(f"{cfg['role']}  ·  Q{cq+1}/{total}")

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#c6ff4e;letter-spacing:.12em;">
        QUESTION {str(cq+1).zfill(2)} / {str(total).zfill(2)}
      </span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:rgba(240,236,228,.3);">
        {pct}% complete
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)

    tags = [cfg["role"], cfg["interview_type"], cfg["difficulty"], cfg["focus_area"]]
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    if rs.get("current_role"):
        tags_html += f'<span class="tag" style="border-color:rgba(183,148,244,.3);color:rgba(183,148,244,.7);">📄 Resume-Aware</span>'
    st.markdown(f'<div style="margin:14px 0 22px;">{tags_html}</div>', unsafe_allow_html=True)

    # Generate question
    if not S.current_question:
        with st.spinner("Generating question…"):
            try:
                q, src = gen_question(cfg, S.questions, cq + 1, rs)
                S.current_question = q
                S.current_q_source = src
                S.voice_answer = ""
                S.current_bl = None
            except Exception as e:
                st.error(f"Failed to generate question: {e}")
                st.stop()

    # Source badge
    if S.current_q_source == "resume":
        badge = '<span class="q-source-badge q-source-resume">📄 Based on your resume</span>'
    else:
        badge = '<span class="q-source-badge q-source-general">⚡ General question</span>'

    st.markdown(f"""
    <div class="q-card">
      <div class="q-kicker">Interviewer · Question {cq+1}</div>
      {badge}
      <div style="margin-top:10px;">{S.current_question}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live integrity alerts ──
    if cfg.get("use_webcam"):
        live_summary = get_final_bl_summary()
        render_alert_banner(live_summary)
    
    detect_tab_switch()


    # TTS
    if cfg.get("auto_tts"):
        announce_question(cq + 1, S.current_question)

    # ── Two-column layout: Answer (left) + Webcam (right) ──
    if cfg.get("use_webcam"):
        col_answer, col_webcam = st.columns([5, 4])
    else:
        col_answer = st.container()
        col_webcam = None

    with col_answer:
        st.markdown('<div class="section-label">Your Answer</div>', unsafe_allow_html=True)

        st.markdown("🎙 **Record your answer:**")
        audio = mic_recorder(
            start_prompt="🎙 Start Recording",
            stop_prompt="⏹ Stop Recording",
            just_once=False,
            use_container_width=True,
            key=f"rec_{cq}"
        )

        if audio:
            audio_id = audio.get("id", 0)
            if audio_id != S.get("last_audio_id", -1):
                S.last_audio_id = audio_id
                with st.spinner("Transcribing with Groq Whisper…"):
                    transcript = transcribe_audio(audio["bytes"], "answer.wav")
                    if not transcript.startswith("[Transcription error"):
                        S.voice_answer = transcript
                        st.success(f"✓ {transcript[:100]}{'…' if len(transcript) > 100 else ''}")
                    else:
                        st.warning(transcript)

        if S.voice_answer:
            col_v, col_c = st.columns([5, 1])
            with col_v:
                st.info(f"📝 Transcript: {S.voice_answer[:120]}{'…' if len(S.voice_answer)>120 else ''}")
            with col_c:
                if st.button("🗑", key=f"clr_{cq}", help="Clear & re-record"):
                    S.voice_answer = ""; S.last_audio_id = -1; st.rerun()

        answer = st.text_area(
            "answer_typed", label_visibility="collapsed",
            placeholder="Speak using the mic above, or type your answer here…",
            value=S.voice_answer, height=150,
        )

    if col_webcam is not None:
        with col_webcam:
            st.markdown('<div class="section-label">Body Language</div>', unsafe_allow_html=True)
            render_webcam_monitor()

    # ── Submit / Skip ──
    st.write("")
    col_sub, col_skip = st.columns([5, 1])
    with col_sub:
        submit = st.button("Submit Answer  →", type="primary", use_container_width=True, key=f"sub_{cq}")
    with col_skip:
        skip = st.button("Skip", type="secondary", use_container_width=True, key=f"skip_{cq}")

    if submit:
        final_answer = answer.strip()
        if not final_answer:
            st.warning("Please speak or type your answer before submitting.")
        else:
            with st.spinner("Evaluating your answer with LLaMA 3.3…"):
                try:
                    fb = gen_feedback(cfg, S.current_question, final_answer, cq + 1, rs)
                    S.questions.append(S.current_question)
                    S.answers.append(final_answer)
                    S.feedbacks.append(fb)
                    frames = st.session_state.get("bl_frames", [])
                    if frames:
                        averaged_bl = {
                            "eye_contact":      round(sum(f["eye_contact"] for f in frames) / len(frames), 1),
                            "confidence":       round(sum(f["confidence"] for f in frames) / len(frames), 1),
                            "posture":          round(sum(f["posture"] for f in frames) / len(frames), 1),
                            "overall_presence": round(sum(f["overall_presence"] for f in frames) / len(frames), 1),
                            "expression":       frames[-1].get("expression", ""),
                            "observations":     frames[-1].get("observations", []),
                            "tip":              frames[-1].get("tip", ""),
                        }
                        S.body_lang_results.append(averaged_bl)
                    else:
                        bl_result = get_final_bl_summary()
                        S.body_lang_results.append(bl_result)
                        # NOTE: Do NOT reset_analyzer() here — keeps stream alive
                    st.session_state.bl_frames = []
                    S.current_feedback = fb
                    S.voice_answer = ""
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse AI feedback. Try again. ({e})")
                except Exception as e:
                    st.error(f"Feedback error: {e}")

    if skip:
        S.questions.append(S.current_question)
        S.answers.append("[skipped]")
        S.feedbacks.append(None)
        S.body_lang_results.append(None)
        S.current_feedback = None; S.current_question = ""
        S.voice_answer = ""; S.current_bl = None
        S.screen = "summary" if (S.current_q + 1 >= total) else "interview"
        if S.screen == "interview":
            S.current_q += 1
        # NOTE: No reset_analyzer() — keep stream alive across questions
        st.rerun()

    # ── Feedback Display ──
    if S.current_feedback:
        fb = S.current_feedback
        kw_html  = "".join(f'<span class="kw-hit">✓ {k}</span>' for k in (fb.get("keywords_mentioned") or []))
        kw_html += "".join(f'<span class="kw-miss">✗ {k}</span>' for k in (fb.get("keywords_missing") or []))
        tip_html = f'<div class="fb-tip">💡 {fb["tip"]}</div>' if fb.get("tip") else ""
        kw_block = f'<div style="margin-top:14px;">{kw_html}</div>' if kw_html else ""

        st.markdown(f"""
        <div class="fb-card">
          <div class="fb-kicker">AI Evaluation</div>
          <div class="fb-text">{fb["feedback"]}</div>
          {tip_html}{kw_block}
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Clarity",  f"{fb['clarity']}/10")
        m2.metric("Depth",    f"{fb['depth']}/10")
        m3.metric("Overall",  f"{fb['overall']}/10")

        st.write("")
        is_last = (cq + 1 >= total)
        btn_label = "View Results  →" if is_last else "Next Question  →"
        if st.button(btn_label, type="primary", use_container_width=True, key=f"next_{cq}"):
            S.current_feedback = None; S.current_question = ""
            S.voice_answer = ""; S.current_bl = None
            S.screen = "summary" if is_last else "interview"
            if not is_last:
                S.current_q += 1
            st.rerun()

        if S.questions:
            with st.expander(f"📋 Previous answers ({len(S.questions)})"):
                for i, q in enumerate(S.questions):
                    f2 = S.feedbacks[i]
                    sc = f2["overall"] if f2 else None
                    sc_cls = score_class(sc)
                    sc_txt = f"{sc}/10" if sc is not None else "–"
                    fb_txt = (f2["feedback"][:100] + "…") if f2 else "Skipped"
                    q_short = (q[:90] + "…") if len(q) > 90 else q
                    st.markdown(f"""
                    <div class="hist-item">
                      <div class="{sc_cls}">{sc_txt}</div>
                      <div>
                        <div class="hist-q">{q_short}</div>
                        <div class="hist-fb">{fb_txt}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif S.screen == "summary":
    cfg    = S.cfg
    scored = [f for f in S.feedbacks if f]
    rs     = S.resume_summary if isinstance(S.resume_summary, dict) else {}

    render_nav("Results")

    st.markdown(f"""
    <h2 style="font-family:'Playfair Display',serif;font-size:clamp(32px,5vw,52px);
               line-height:1.1;margin-bottom:8px;color:#f0ece4;">
      Interview<br/><em style="color:#c6ff4e;">Complete.</em>
    </h2>
    <p style="color:rgba(240,236,228,.48);font-size:14px;margin-bottom:32px;">
      {cfg['role']}  ·  {cfg['interview_type']}  ·  {cfg['difficulty']}
      &nbsp;—&nbsp; {len(scored)} of {cfg['total_questions']} questions answered
      {f"&nbsp;·&nbsp; Resume: {rs.get('current_role','')}" if rs.get('current_role') else ''}
    </p>
    """, unsafe_allow_html=True)

    if scored:
        ac = avg([f["clarity"] for f in scored])
        ad = avg([f["depth"]   for f in scored])
        ao = avg([f["overall"] for f in scored])

        bl_scored = [b for b in S.body_lang_results if b and b.get("overall_presence")]
        bl_avg = avg([b["overall_presence"] for b in bl_scored])
        bl_ec  = avg([b["eye_contact"] for b in bl_scored])
        bl_cf  = avg([b["confidence"]  for b in bl_scored])

        if bl_scored:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Clarity",    f"{ac}/10")
            m2.metric("Depth",      f"{ad}/10")
            m3.metric("Overall",    f"{ao}/10")
            m4.metric("Eye Contact",f"{bl_ec}/10")
            m5.metric("Presence",   f"{bl_avg}/10")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Clarity", f"{ac}/10")
            m2.metric("Depth",   f"{ad}/10")
            m3.metric("Overall", f"{ao}/10")
    else:
        st.info("No answers were submitted this session.")

    # AI Summary
    if not S.summary_text:
        with st.spinner("Generating AI performance assessment…"):
            try:
                S.summary_text = gen_summary(
                    cfg, S.questions, S.answers, S.feedbacks,
                    S.body_lang_results, rs
                )
            except Exception as e:
                S.summary_text = f"Could not generate summary: {e}"

    st.markdown(f"""
    <div class="assess-box">
      <div class="assess-kicker">AI Performance Assessment</div>
      <div class="assess-text">{S.summary_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Voice summary — spoken once per session
    if scored and not S.get("summary_announced"):
        announce_summary(S.summary_text)
        S.summary_announced = True

    # ── Aggregate integrity alerts ──
    if S.body_lang_results:
        all_events, total_phone, total_multi, total_absent = [], 0, 0, 0

        for bl in S.body_lang_results:
            if bl:
                all_events.extend(bl.get("cheat_events", []))
                total_phone  += bl.get("phone_events", 0)
                total_multi  += bl.get("multiple_face_events", 0)
                total_absent += bl.get("face_absent_frames", 0)

        render_alert_banner({
            "cheat_score": max(
                (bl.get("cheat_score", 0) for bl in S.body_lang_results if bl),
                default=0
            ),
            "cheat_events": all_events,
            "phone_events": total_phone,
            "multiple_face_events": total_multi,
            "face_absent_frames": total_absent,
        })
        

    # Export / New
    col_pdf, col_new = st.columns(2)
    with col_pdf:
        if scored and PDF_WRITE_AVAILABLE:
            try:
                pdf_buf = build_pdf(
                    cfg, S.questions, S.answers, S.feedbacks,
                    S.body_lang_results, S.summary_text, rs
                )
                st.download_button(
                    label="⬇  Export Full PDF Report",
                    data=pdf_buf,
                    file_name=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"PDF error: {e}")
        else:
            reason = "No answers" if not scored else "reportlab missing"
            st.button(f"⬇  Export PDF ({reason})", disabled=True, use_container_width=True)
    with col_new:
        if st.button("New Interview  →", type="secondary", use_container_width=True):
            for k in ["screen","cfg","resume_text","resume_summary","questions","answers",
                      "feedbacks","body_lang_results","current_q","current_question",
                      "current_q_source","current_feedback","current_bl","summary_text",
                      "voice_answer","last_audio_id","summary_announced"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Question Breakdown</div>', unsafe_allow_html=True)

    for i, q in enumerate(S.questions):
        f   = S.feedbacks[i]
        bl  = S.body_lang_results[i] if i < len(S.body_lang_results) else None
        sc  = f["overall"] if f else None
        sc_cls = score_class(sc)
        sc_txt = f"{sc}/10" if sc is not None else "–"
        fb_txt = (f["feedback"][:130] + "…") if f and len(f["feedback"]) > 130 else (f["feedback"] if f else "Skipped")
        q_short = (q[:110] + "…") if len(q) > 110 else q

        bl_extra = ""
        if bl and bl.get("overall_presence"):
            bl_extra = f' <span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:rgba(90,176,255,.6);margin-left:8px;">👁 {bl["overall_presence"]}/10</span>'

        st.markdown(f"""
        <div class="hist-item">
          <div class="{sc_cls}">{sc_txt}{bl_extra}</div>
          <div>
            <div class="hist-q">Q{i+1}. {q_short}</div>
            <div class="hist-fb">{fb_txt}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if f:
            with st.expander(f"Full feedback — Q{i+1}"):
                kw_html  = "".join(f'<span class="kw-hit">✓ {k}</span>' for k in (f.get("keywords_mentioned") or []))
                kw_html += "".join(f'<span class="kw-miss">✗ {k}</span>' for k in (f.get("keywords_missing") or []))
                tip_html = f'<div class="fb-tip" style="margin-top:10px;">💡 {f["tip"]}</div>' if f.get("tip") else ""
                st.markdown(f"""
                <div class="fb-card" style="margin:0;">
                  <div class="fb-text">{f['feedback']}</div>
                  {tip_html}
                  <div style="margin-top:12px;">{kw_html}</div>
                </div>
                """, unsafe_allow_html=True)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Clarity", f"{f['clarity']}/10")
                mc2.metric("Depth",   f"{f['depth']}/10")
                mc3.metric("Overall", f"{f['overall']}/10")

                if bl and bl.get("overall_presence"):
                    bm1, bm2, bm3, bm4 = st.columns(4)
                    bm1.metric("Eye Contact", f"{bl['eye_contact']}/10")
                    bm2.metric("Confidence",  f"{bl['confidence']}/10")
                    bm3.metric("Posture",     f"{bl['posture']}/10")
                    bm4.metric("Presence",    f"{bl['overall_presence']}/10")
                    if bl.get("tip"):
                        st.info(f"👁 Body language tip: {bl['tip']}")

                if S.answers[i] and S.answers[i] != "[skipped]":
                    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:rgba(240,236,228,.3);letter-spacing:.12em;text-transform:uppercase;margin-top:16px;">Your Answer</p>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:13px;color:rgba(240,236,228,.6);line-height:1.8;background:#181825;border-radius:10px;padding:14px 18px;">{S.answers[i]}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:48px;padding-top:24px;border-top:1px solid rgba(240,236,228,.06);">
      <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:rgba(240,236,228,.2);letter-spacing:.1em;">
        INTERVIEWAI · RESUME-AWARE EDITION · GROQ + LLAMA 3.3 + WHISPER + VISION
      </span>
    </div>
    """, unsafe_allow_html=True)
