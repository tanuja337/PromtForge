"""
PromptForge — AI-Powered Prompt Evaluation Studio
Run with: streamlit run app.py
"""
from __future__ import annotations

import csv
import difflib
import html
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

APP_DIR = Path(__file__).parent
HISTORY_FILE = APP_DIR / "history.json"
MODEL_NAME = "openai/gpt-oss-20b"
WEIGHTS = {
    "Role": 10,
    "Goal": 20,
    "Context": 20,
    "Constraints": 15,
    "Format": 15,
    "Style": 10,
}
DIMENSIONS = list(WEIGHTS.keys())

SCENARIOS = [
    {"id": "CS-B1", "title": "Reply to a delayed delivery complaint", "domain": "Customer Support", "difficulty": "Beginner",
     "situation": "A customer is upset because an online order has not arrived on the promised date.",
     "requirements": ["Acknowledge the customer's frustration", "Apologize professionally", "Explain the next step without inventing tracking facts", "Keep the reply concise and helpful"],
     "hint": "Ask for a calm, empathetic customer email with a clear next step."},
    {"id": "CS-I1", "title": "Handle a refund request", "domain": "Customer Support", "difficulty": "Intermediate",
     "situation": "A customer requests a refund for a product that may be outside the normal return window.",
     "requirements": ["Ask for relevant order details", "Explain policy carefully without inventing policy", "Offer a helpful escalation path", "Use empathetic language"],
     "hint": "Specify what information is known, what must be checked, and the response format."},
    {"id": "CS-A1", "title": "De-escalate a high-priority service issue", "domain": "Customer Support", "difficulty": "Advanced",
     "situation": "A business customer reports a critical service interruption affecting their team.",
     "requirements": ["Prioritize impact and urgency", "Avoid unsupported promises", "Request essential diagnostic information", "Provide a structured update and escalation plan"],
     "hint": "Define the support role, triage process, boundaries, and update format."},
    {"id": "M-B1", "title": "Write a product social caption", "domain": "Marketing", "difficulty": "Beginner",
     "situation": "A small business is launching a reusable water bottle.",
     "requirements": ["Identify the target audience", "Highlight two supplied product benefits", "Include a clear call to action", "Use an engaging but natural tone"],
     "hint": "Ask for a platform-ready caption and specify the audience and tone."},
    {"id": "M-I1", "title": "Create a campaign brief", "domain": "Marketing", "difficulty": "Intermediate",
     "situation": "A team needs a one-week campaign to increase sign-ups for a free webinar.",
     "requirements": ["Define campaign objective and audience", "Suggest a one-week content sequence", "Include channels and calls to action", "State assumptions clearly"],
     "hint": "Set measurable goals, audience context, deliverable structure, and constraints."},
    {"id": "M-A1", "title": "Analyze a campaign performance summary", "domain": "Marketing", "difficulty": "Advanced",
     "situation": "A marketing manager needs insights from campaign metrics and recommendations for the next month.",
     "requirements": ["Separate observations from assumptions", "Identify trends and possible causes", "Recommend prioritized experiments", "Use a concise executive-report format"],
     "hint": "Tell the AI how to reason from supplied data and avoid unsupported causal claims."},
    {"id": "HR-B1", "title": "Draft an interview invitation", "domain": "HR", "difficulty": "Beginner",
     "situation": "A recruiter needs to invite a candidate to a first-round interview.",
     "requirements": ["Use a professional welcoming tone", "Include placeholders for date and time", "Mention interview format", "Explain how to confirm availability"],
     "hint": "Specify audience, details to include, and an email format."},
    {"id": "HR-I1", "title": "Build an onboarding checklist", "domain": "HR", "difficulty": "Intermediate",
     "situation": "HR wants a practical first-week onboarding checklist for a new employee.",
     "requirements": ["Organize tasks by day or category", "Include owner or responsible role", "Cover tools, people, and policies", "Keep it actionable"],
     "hint": "Define employee context, timeframe, sections, and checklist format."},
    {"id": "HR-A1", "title": "Create structured performance feedback", "domain": "HR", "difficulty": "Advanced",
     "situation": "A manager needs balanced feedback based on documented performance examples.",
     "requirements": ["Use only supplied evidence", "Balance strengths and development areas", "Avoid biased or personal assumptions", "Set specific next steps"],
     "hint": "Add evidence boundaries, fairness constraints, tone, and a structured format."},
    {"id": "DS-B1", "title": "Explain a Python function", "domain": "Data & Software", "difficulty": "Beginner",
     "situation": "A beginner wants to understand a short Python function.",
     "requirements": ["Explain in beginner-friendly language", "Describe inputs and outputs", "Walk through the main steps", "Include a small example"],
     "hint": "Assign a teaching role and define the learner level and explanation format."},
    {"id": "DS-I1", "title": "Debug a data-cleaning issue", "domain": "Data & Software", "difficulty": "Intermediate",
     "situation": "A data analyst has inconsistent date values in a CSV file.",
     "requirements": ["Ask for a representative sample if needed", "Explain likely causes", "Suggest safe pandas approaches", "Include validation checks"],
     "hint": "Provide environment context, expected behavior, constraints, and output structure."},
    {"id": "DS-A1", "title": "Design a small analytics pipeline", "domain": "Data & Software", "difficulty": "Advanced",
     "situation": "A team needs a maintainable pipeline to ingest, validate, transform, and summarize CSV data.",
     "requirements": ["Define pipeline stages", "Include validation and error handling", "Discuss maintainability and assumptions", "Specify testable deliverables"],
     "hint": "Set an engineering role, scope, constraints, testing expectations, and design-document format."},
    {"id": "F-B1", "title": "Explain a monthly budget", "domain": "Finance", "difficulty": "Beginner",
     "situation": "A student wants a simple explanation of how to organize monthly spending.",
     "requirements": ["Use accessible language", "Separate needs, wants, and savings", "Avoid assuming income details", "Provide a simple example"],
     "hint": "Define the audience, context, tone, and table or bullet format."},
    {"id": "F-I1", "title": "Summarize expense categories", "domain": "Finance", "difficulty": "Intermediate",
     "situation": "A small business owner wants a summary of expense data they will provide.",
     "requirements": ["Group expenses into clear categories", "Calculate totals only from supplied data", "Flag ambiguous entries", "Present a concise summary table"],
     "hint": "Set data boundaries, calculations, treatment of ambiguity, and output format."},
    {"id": "F-A1", "title": "Prepare a financial scenario comparison", "domain": "Finance", "difficulty": "Advanced",
     "situation": "A team needs to compare two hypothetical operating-cost scenarios.",
     "requirements": ["State assumptions", "Compare like-for-like cost drivers", "Show calculations transparently", "Avoid presenting the result as personalized financial advice"],
     "hint": "Specify inputs, assumptions, comparison criteria, calculation transparency, and limitations."},
]

st.set_page_config(page_title="PromptForge | Prompt Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(0,210,255,0.12), transparent 30%),
        radial-gradient(circle at 85% 75%, rgba(130,80,255,0.14), transparent 32%),
        linear-gradient(135deg, #080D1B, #10182D, #080D1B);
    color: #F1F5F9;
}

/* Animated AI particle effect */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.35;
    background-image:
        radial-gradient(circle, #38BDF8 1.5px, transparent 2px),
        radial-gradient(circle, #A78BFA 1px, transparent 2px),
        radial-gradient(circle, #22D3EE 1px, transparent 2px);
    background-size: 95px 95px, 135px 135px, 175px 175px;
    background-position: 0 0, 45px 65px, 90px 30px;
    animation: aiParticles 35s linear infinite;
}

@keyframes aiParticles {
    from {
        background-position: 0 0, 45px 65px, 90px 30px;
    }
    to {
        background-position: 300px 500px, -250px 350px, 200px -300px;
    }
}

/* Professional headings */
h1, h2, h3 {
    color: #E0F2FE !important;
    font-weight: 750 !important;
}

/* Cards */
div[data-testid="stMetric"],
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(20, 35, 60, 0.72);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 15px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(100deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 9px 20px;
    font-weight: 650;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(56,189,248,0.3);
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div {
    border-radius: 9px;
}

/* Respect reduced-motion settings */
@media (prefers-reduced-motion: reduce) {
    .stApp::before {
        animation: none;
    }
}

</style>
""", unsafe_allow_html=True)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --surface: #111a2b;
    --surface2: #17233a;
    --purple: #9675ff;
    --cyan: #51d9ed;
    --text: #f4f5ff;
    --muted: #9baac5;
    --border: rgba(159,135,255,.20);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        -45deg,
        #080b18,
        #171038,
        #101e3d,
        #251044,
        #09182c,
        #080b18
    );
    background-size: 400% 400%;
    animation: pfGradientMove 18s ease infinite;
    color: #f5f5ff;
}

@keyframes pfGradientMove {
    0% {
        background-position: 0% 50%;
    }
    25% {
        background-position: 50% 100%;
    }
    50% {
        background-position: 100% 50%;
    }
    75% {
        background-position: 50% 0%;
    }
    100% {
        background-position: 0% 50%;
    }
}
@media (prefers-reduced-motion: reduce) {
    .stApp {
        animation: none !important;
        background-position: 50% 50%;
    }

    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
    }
}

.block-container {
    max-width: 1500px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -.035em;
    color: #f5f5ff;
}

p, label, .stCaption {
    color: #b4c0d8;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#121c31,#090f1e);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #aab8d2;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#192641,#111a2c);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,.18);
    transition: all .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(160,130,255,.65);
    box-shadow: 0 14px 35px rgba(90,70,180,.16);
}

[data-testid="stMetricLabel"] {
    color: #aebbd6 !important;
    font-size: .88rem;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
}

/* BUTTONS */
div.stButton > button,
div.stDownloadButton > button {
    border-radius: 12px;
    min-height: 2.9rem;
    border: 1px solid rgba(164,143,255,.35);
    background: linear-gradient(110deg,#7051e8,#9870ff);
    color: white;
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: #c8baff;
    color: white;
    box-shadow: 0 7px 24px rgba(125,91,255,.28);
    transform: translateY(-2px);
}

/* INPUTS */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #111b30;
    color: white;
    border: 1px solid #344565;
    border-radius: 12px;
}

div[data-testid="stSelectbox"] > div > div {
    background: #111b30;
    border-radius: 10px;
}

div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #9675ff;
    box-shadow: 0 0 0 1px #9675ff;
}

/* CONTAINERS */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 18px !important;
}

/* HERO SECTION */
.pf-hero {
    position: relative;
    overflow: hidden;
    padding: 24px 30px;
    border: 1px solid rgba(170,145,255,.32);
    border-radius: 22px;
    background:
        radial-gradient(circle at 90% 10%,rgba(91,204,255,.22),transparent 30%),
        radial-gradient(circle at 5% 100%,rgba(142,92,255,.25),transparent 40%),
        linear-gradient(120deg,#202b50,#34265e 55%,#172c49);
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
    margin-bottom: 18px;
}

.pf-kicker {
    color: #c0adff;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .17em;
    text-transform: uppercase;
}

.pf-hero-title {
    font-family: 'Space Grotesk',sans-serif;
    font-size: clamp(2rem,4vw,3rem);
    font-weight: 700;
    line-height: 1.15;
    color: white;
    margin: 14px 0;
}

.pf-hero-sub {
    color: #d0d9f0;
    font-size: .98rem;
    max-width: 720px;
    line-height: 1.8;
}

/* BADGES */
.pf-pill {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(158,126,255,.17);
    color: #ded3ff;
    border: 1px solid rgba(180,157,255,.30);
    font-size: .75rem;
    font-weight: 700;
}

/* SECTION HEADINGS */
.pf-section {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f4f5ff;
    margin-top: 25px;
    margin-bottom: 7px;
}

.pf-subtitle {
    color: #9baac5;
    font-size: .87rem;
    margin-bottom: 17px;
    line-height: 1.6;
}

/* INFORMATION CARDS */
.pf-panel {
    background: linear-gradient(145deg,#182541,#10192c);
    border: 1px solid rgba(154,137,255,.22);
    border-radius: 19px;
    padding: 23px;
    min-height: 135px;
    box-shadow: 0 8px 25px rgba(0,0,0,.13);
    transition: all .25s ease;
}

.pf-panel:hover {
    border-color: rgba(157,130,255,.55);
    transform: translateY(-3px);
}

.pf-panel-title {
    font-weight: 700;
    color: #f0efff;
    font-size: 1.02rem;
}

.pf-panel-desc {
    color: #a4b2ce;
    font-size: .85rem;
    line-height: 1.7;
    margin-top: 9px;
}

/* EMPTY DASHBOARD */
.pf-empty {
    background:
        radial-gradient(circle at 100% 0%,rgba(135,95,255,.16),transparent 40%),
        linear-gradient(120deg,#17233c,#211c40);
    border: 1px solid rgba(164,143,255,.28);
    border-radius: 22px;
    padding: 30px;
}

/* TABLES */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(154,137,255,.24);
    border-radius: 15px;
    overflow: hidden;
}

/* CHARTS */
div[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg,rgba(22,33,55,.55),rgba(14,23,40,.35));
    border: 1px solid rgba(154,137,255,.13);
    border-radius: 17px;
    padding: 8px;
}

/* PROGRESS */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#7958ed,#56d7e9);
    border-radius: 20px;
}

/* ALERTS */
div[data-testid="stAlert"] {
    border-radius: 13px;
}

/* DIVIDERS */
hr {
    border-color: rgba(154,137,255,.18);
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0b1120;
}

::-webkit-scrollbar-thumb {
    background: #51427f;
    border-radius: 10px;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .pf-hero {
        padding: 23px;
    }

    .pf-hero-title {
        font-size: 1.9rem;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def load_history() -> list[dict[str, Any]]:
    if "history" in st.session_state:
        return st.session_state.history
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            st.session_state.history = data if isinstance(data, list) else []
        else:
            st.session_state.history = []
    except (OSError, json.JSONDecodeError):
        st.session_state.history = []
    return st.session_state.history

def save_history(history: list[dict[str, Any]]) -> None:
    st.session_state.history = history
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        st.warning(f"Could not persist history to disk: {exc}")

def get_client() -> Groq:
    load_dotenv(APP_DIR / ".env", override=False)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("Groq API key is missing. Add your key to the .env file as GROQ_API_KEY=...")
    return Groq(api_key=api_key)

def build_judge_messages(prompt: str, scenario: dict[str, Any] | None) -> list[dict[str, str]]:
    scenario_text = "General prompt evaluation (no selected scenario)." if not scenario else (
        f"Scenario: {scenario['title']} | Domain: {scenario['domain']} | Difficulty: {scenario['difficulty']}\n"
        f"Situation: {scenario['situation']}\nRequirements: {json.dumps(scenario['requirements'])}\n"
        f"Expected-output hint: {scenario['hint']}"
    )
    system = f"""You are a rigorous, fair prompt-engineering evaluator and coach. Evaluate only the prompt supplied by the user, not an imagined completed answer. A dimension can receive a low score or zero when the prompt does not provide evidence. Do not reward information that is absent. Do not require every prompt to have all six elements if the task genuinely does not need them, but assess clarity and usefulness. If a scenario is supplied, evaluate fit against its requirements.
Return ONLY valid JSON with this exact schema:
{{
 "scores": {{"Role": 0, "Goal": 0, "Context": 0, "Constraints": 0, "Format": 0, "Style": 0}},
 "justifications": {{"Role": "one sentence citing brief exact evidence or stating what is missing", "Goal": "...", "Context": "...", "Constraints": "...", "Format": "...", "Style": "..."}},
 "strengths": ["...", "...", "..."],
 "weaknesses": ["...", "...", "..."],
 "recommendations": ["...", "...", "..."],
 "scenario_fit": "brief, specific fit assessment",
 "optimized_prompt": "complete improved prompt preserving the user's intent",
 "changes": ["change and why", "change and why", "change and why"]
}}
All scores must be numeric from 0 to 10. Give specific, evidence-grounded feedback. Do not invent facts or claim that missing instructions exist. Recommendations should be actionable. The optimized prompt must preserve the original objective and must not introduce unsupported facts."""
    user = f"""Evaluate this submitted prompt.
{scenario_text}

USER PROMPT:
<user_prompt>
{prompt}
</user_prompt>"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The AI response was not valid JSON.")
        obj = json.loads(cleaned[start:end + 1])

    if not isinstance(obj, dict):
        raise ValueError("The AI response must be a JSON object.")

    required_keys = [
        "scores", "justifications", "strengths",
        "weaknesses", "recommendations", "scenario_fit",
        "optimized_prompt", "changes"
    ]

    missing = [key for key in required_keys if key not in obj]
    if missing:
        raise ValueError(f"AI response is missing: {', '.join(missing)}")

    scores = obj["scores"]
    justifications = obj["justifications"]

    if not isinstance(scores, dict) or not isinstance(justifications, dict):
        raise ValueError("Scores and justifications must be objects.")

    normalized_scores = {}
    normalized_justifications = {}

    for dim in DIMENSIONS:
        value = scores.get(dim)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid numeric score for {dim}.")

        if not 0 <= value <= 10:
            raise ValueError(f"{dim} score must be between 0 and 10.")

        normalized_scores[dim] = float(value)

        explanation = justifications.get(dim)
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Missing explanation for {dim}.")

        normalized_justifications[dim] = explanation.strip()

    def clean_list(key: str) -> list[str]:
        value = obj[key]

        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} must contain at least one item.")

        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{key} contains an invalid item.")

        return [item.strip() for item in value[:3]]

    scenario_fit = obj["scenario_fit"]
    optimized_prompt = obj["optimized_prompt"]

    if not isinstance(scenario_fit, str) or not scenario_fit.strip():
        raise ValueError("Scenario fit feedback is missing.")

    if not isinstance(optimized_prompt, str) or not optimized_prompt.strip():
        raise ValueError("Optimized prompt is missing.")

    return {
        "scores": normalized_scores,
        "justifications": normalized_justifications,
        "strengths": clean_list("strengths"),
        "weaknesses": clean_list("weaknesses"),
        "recommendations": clean_list("recommendations"),
        "scenario_fit": scenario_fit.strip(),
        "optimized_prompt": optimized_prompt.strip(),
        "changes": clean_list("changes"),
    }

def evaluate_prompt(prompt: str, scenario: dict[str, Any] | None) -> dict[str, Any]:
    client = get_client()
    messages = build_judge_messages(prompt, scenario)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return parse_judge_json(raw)
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                messages = messages + [{"role": "user", "content": "Your previous response could not be parsed or validated. Return the complete response again as valid JSON matching the exact schema. Do not include markdown."}]
    raise RuntimeError(f"Evaluation failed after one retry: {last_error}")

def overall_score(scores: dict[str, float]) -> float:
    weighted_sum = sum(scores[d] * WEIGHTS[d] for d in DIMENSIONS)
    return round((weighted_sum / sum(WEIGHTS.values())) * 10, 1)

def skill_level(score: float) -> str:
    if score >= 90: return "Master"
    if score >= 75: return "Expert"
    if score >= 55: return "Practitioner"
    return "Novice"

def make_radar(scores: dict[str, float]) -> go.Figure:
    values = [scores[d] for d in DIMENSIONS]
    fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=DIMENSIONS + [DIMENSIONS[0]],
        fill="toself", fillcolor="rgba(139,92,246,.22)", line=dict(color="#a78bfa", width=3),
        marker=dict(size=7, color="#c4b5fd")))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0,10], tickfont=dict(color="#aab5d1"), gridcolor="#303b57"),
        angularaxis=dict(tickfont=dict(color="#e5e7eb"), gridcolor="#303b57")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", margin=dict(l=35,r=35,t=25,b=25), height=330, showlegend=False)
    return fig

def make_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix":"/100","font":{"size":36,"color":"#f5f3ff"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#aab5d1"}, "bar":{"color":"#a78bfa","thickness":.28},
               "bgcolor":"#1b2540","borderwidth":0,"steps":[{"range":[0,50],"color":"#442b3c"},{"range":[50,75],"color":"#4a402b"},{"range":[75,100],"color":"#24483f"}]}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=245, margin=dict(l=25,r=25,t=30,b=10))
    return fig

def render_metric_row(history: list[dict[str, Any]]) -> None:
    avg = round(sum(x["overall"] for x in history) / len(history), 1) if history else 0
    best = max((x["overall"] for x in history), default=0)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Evaluations", len(history))
    c2.metric("Average score", f"{avg}/100")
    c3.metric("Personal best", f"{best}/100")
    c4.metric("Current level", skill_level(history[-1]["overall"]) if history else "Not rated")

def render_dashboard(history: list[dict[str, Any]]) -> None:
    # Safe, chronological history
    history = sorted(history, key=lambda x: x.get("timestamp", ""))

    total = len(history)
    scores = [float(x.get("overall", 0)) for x in history]
    avg = round(sum(scores) / total, 1) if total else 0
    best = max(scores, default=0)
    latest = history[-1] if history else None
    previous = history[-2] if total > 1 else None

    improvement = (
        round(scores[-1] - scores[-2], 1)
        if total > 1 else None
    )

    # Hero
    st.markdown("""
    <div class="pf-hero">
        <div class="pf-kicker">✦ PROMPTFORGE / AI INTELLIGENCE PLATFORM</div>
        <div class="pf-hero-title">Your AI Command Center</div>
        <div class="pf-hero-sub">
            Measure prompt quality. Discover hidden weaknesses.
            Track your growth and turn better instructions into better AI results.
        </div>
        <br>
        <span class="pf-pill">✦ LLM-as-a-Judge</span>
        &nbsp;
        <span class="pf-pill">◈ Six-dimension analysis</span>
        &nbsp;
        <span class="pf-pill">⚡ Powered by Groq</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown('<div class="pf-section">Performance overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pf-subtitle">Your real evaluation statistics at a glance.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "◈ Total evaluations",
        total,
        help="Number of successfully completed prompt evaluations."
    )

    c2.metric(
        "◎ Average quality",
        f"{avg}/100",
        help="Average overall score across your evaluations."
    )

    c3.metric(
        "🏆 Personal best",
        f"{best}/100",
        help="Your highest recorded prompt score."
    )

    c4.metric(
        "↗ Latest improvement",
        f"{improvement:+.1f}" if improvement is not None else "—",
        help="Difference between your latest two evaluation scores."
    )

    # First-time dashboard
    if not history:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="pf-empty">
            <div class="pf-kicker">YOUR JOURNEY STARTS HERE</div>
            <h2 style="color:white;margin-top:10px;">
                Ready to engineer better prompts?
            </h2>
            <p style="color:#bac7e4;line-height:1.8;">
                Your analytics centre is ready. Complete your first AI
                evaluation to unlock performance charts, skill analysis,
                personalized insights and your improvement journey.
            </p>
            <span class="pf-pill">01 · Evaluate</span>
            &nbsp;
            <span class="pf-pill">02 · Discover</span>
            &nbsp;
            <span class="pf-pill">03 · Improve</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Explore your workspace")

        a, b, c = st.columns(3)

        with a:
            st.markdown("""
            <div class="pf-panel">
                <div class="pf-panel-title">✦ Prompt Evaluation</div>
                <div class="pf-panel-desc">
                    Analyze prompts across six engineering dimensions
                    and receive evidence-based feedback.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start evaluating →", key="empty_eval",
                         use_container_width=True):
                st.session_state.page = "Practice & Evaluate"
                st.rerun()

        with b:
            st.markdown("""
            <div class="pf-panel">
                <div class="pf-panel-title">⚡ Prompt Optimizer</div>
                <div class="pf-panel-desc">
                    Transform unclear instructions into structured,
                    specific and effective prompts.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Explore optimizer →", key="empty_opt",
                         use_container_width=True):
                st.session_state.page = "Prompt Optimizer"
                st.rerun()

        with c:
            st.markdown("""
            <div class="pf-panel">
                <div class="pf-panel-title">◈ Progress Analytics</div>
                <div class="pf-panel-desc">
                    Monitor your scores, identify weak dimensions
                    and follow your learning journey.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View progress →", key="empty_progress",
                         use_container_width=True):
                st.session_state.page = "Progress Tracker"
                st.rerun()

        return

    # Latest evaluation and skill
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="pf-section">Latest performance</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="pf-subtitle">Your most recent prompt evaluation.</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            make_gauge(float(latest.get("overall", 0))),
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_latest_gauge"
        )

        level = skill_level(float(latest.get("overall", 0)))
        st.markdown(
            f'<span class="pf-pill">✦ {level} level</span>',
            unsafe_allow_html=True
        )

        st.caption(
            f"Latest evaluation: {latest.get('timestamp', 'Date unavailable')}"
        )

    with right:
        st.markdown('<div class="pf-section">Your skill profile</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="pf-subtitle">Performance across six dimensions.</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            make_radar(latest.get("scores", {})),
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_latest_radar"
        )

    # Improvement journey
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pf-section">Your improvement journey</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pf-subtitle">Track how your prompt quality changes over time.</div>',
        unsafe_allow_html=True
    )

    trend = pd.DataFrame([
        {
            "Attempt": i + 1,
            "Score": float(item.get("overall", 0)),
            "Date": item.get("timestamp", "")[:10],
            "Scenario": item.get("scenario_title", "General")
        }
        for i, item in enumerate(history)
    ])

    fig = px.area(
        trend,
        x="Attempt",
        y="Score",
        markers=True,
        hover_data=["Date", "Scenario"],
        range_y=[0, 100]
    )

    fig.update_traces(
        line=dict(color="#a78bfa", width=3),
        fillcolor="rgba(139,92,246,.15)",
        marker=dict(size=8, color="#c4b5fd")
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        height=320,
        margin=dict(l=15, r=20, t=20, b=15),
        xaxis=dict(
            title="Evaluation attempt",
            gridcolor="rgba(130,145,180,.12)"
        ),
        yaxis=dict(
            title="Prompt quality score",
            range=[0, 100],
            gridcolor="rgba(130,145,180,.18)"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="dashboard_trend"
    )

    # Dimension analytics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pf-section">Prompt intelligence analytics</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pf-subtitle">Understand which parts of your prompt engineering need attention.</div>',
        unsafe_allow_html=True
    )

    dimension_rows = []

    for dim in DIMENSIONS:
        values = [
            float(item.get("scores", {}).get(dim, 0))
            for item in history
        ]
        dimension_rows.append({
            "Dimension": dim,
            "Average": round(sum(values) / len(values), 2) if values else 0
        })

    dimension_df = pd.DataFrame(dimension_rows)

    chart_left, chart_right = st.columns([1.15, 1])

    with chart_left:
        st.markdown("#### Average dimension scores")

        bar = px.bar(
            dimension_df.sort_values("Average"),
            x="Average",
            y="Dimension",
            orientation="h",
            range_x=[0, 10],
            text="Average"
        )

        bar.update_traces(
            marker_color="#9174ed",
            texttemplate="%{text:.1f}/10",
            textposition="outside"
        )

        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            height=340,
            margin=dict(l=5, r=45, t=15, b=15),
            xaxis=dict(gridcolor="rgba(130,145,180,.15)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )

        st.plotly_chart(
            bar,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_dimension_bar"
        )

    with chart_right:
        st.markdown("#### Score distribution")

        bands = {
            "Needs work": sum(1 for x in scores if x < 50),
            "Developing": sum(1 for x in scores if 50 <= x < 75),
            "Strong": sum(1 for x in scores if 75 <= x < 90),
            "Excellent": sum(1 for x in scores if x >= 90)
        }

        distribution = pd.DataFrame([
            {"Level": k, "Evaluations": v}
            for k, v in bands.items()
        ])

        distribution = distribution[distribution["Evaluations"] > 0]

        donut = px.pie(
            distribution,
            names="Level",
            values="Evaluations",
            hole=.68
        )

        donut.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(
                line=dict(color="#111a30", width=3)
            )
        )

        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            height=340,
            margin=dict(l=5, r=5, t=15, b=15),
            legend=dict(
                orientation="h",
                y=-.08,
                x=.5,
                xanchor="center"
            )
        )

        st.plotly_chart(
            donut,
            use_container_width=True,
            config={"displayModeBar": False},
            key="dashboard_distribution"
        )

    # AI coaching insights derived from real evaluation feedback
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pf-section">✦ Performance intelligence</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pf-subtitle">Insights calculated from your saved evaluation results.</div>',
        unsafe_allow_html=True
    )

    weakest = dimension_df.sort_values("Average").iloc[0]
    strongest = dimension_df.sort_values("Average", ascending=False).iloc[0]

    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        st.markdown(f"""
        <div class="pf-panel">
            <div class="pf-kicker">FOCUS AREA</div>
            <div class="pf-panel-title">{weakest['Dimension']}</div>
            <div class="pf-panel-desc">
                Your lowest average dimension is
                {weakest['Average']:.1f}/10.
                Give this area extra attention in your next prompt.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with insight2:
        st.markdown(f"""
        <div class="pf-panel">
            <div class="pf-kicker">YOUR STRENGTH</div>
            <div class="pf-panel-title">{strongest['Dimension']}</div>
            <div class="pf-panel-desc">
                Your strongest average dimension is
                {strongest['Average']:.1f}/10.
                Continue using the techniques that work well here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with insight3:
        change_text = (
            f"{improvement:+.1f} points"
            if improvement is not None else "More evaluations needed"
        )
        st.markdown(f"""
        <div class="pf-panel">
            <div class="pf-kicker">RECENT MOMENTUM</div>
            <div class="pf-panel-title">{change_text}</div>
            <div class="pf-panel-desc">
                Change between your two most recent evaluation scores.
                This is a comparison, not a prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recent activity
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pf-section">Recent evaluation activity</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pf-subtitle">Your latest saved prompt evaluations.</div>',
        unsafe_allow_html=True
    )

    recent = history[-5:][::-1]

    recent_df = pd.DataFrame([
        {
            "Date": item.get("timestamp", "")[:16].replace("T", " "),
            "Scenario": item.get("scenario_title", "General evaluation"),
            "Score": f"{float(item.get('overall', 0)):.1f}/100",
            "Level": skill_level(float(item.get("overall", 0)))
        }
        for item in recent
    ])

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True
    )

    # Quick actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pf-section">Continue your journey</div>',
                unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button("✦ Evaluate a new prompt", use_container_width=True):
            st.session_state.page = "Practice & Evaluate"
            st.rerun()

    with q2:
        if st.button("⚡ Improve my prompt", use_container_width=True):
            st.session_state.page = "Prompt Optimizer"
            st.rerun()

    with q3:
        if st.button("◈ Full progress report", use_container_width=True):
            st.session_state.page = "Progress Tracker"
            st.rerun()

def render_feedback(result: dict[str, Any], score: float) -> None:
    st.markdown("### Evaluation report")
    a,b = st.columns([1,2])
    with a:
        st.plotly_chart(make_gauge(score), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<span class="pf-pill">{skill_level(score)}</span>', unsafe_allow_html=True)
    with b:
        st.plotly_chart(make_radar(result["scores"]), use_container_width=True, config={"displayModeBar":False})
    st.markdown("#### Dimension-by-dimension analysis")
    for dim in DIMENSIONS:
        val = result["scores"][dim]
        color = "#ef6464" if val < 5 else "#e9ad4e" if val <= 7 else "#39c995"
        st.markdown(f"**{dim}** · <span style='color:{color};font-weight:700'>{val:.1f}/10</span>", unsafe_allow_html=True)
        st.progress(int(val * 10))
        st.caption(result["justifications"][dim])
    c1,c2,c3 = st.columns(3)
    for col,title,key,icon in [(c1,"Strengths","strengths","✦"),(c2,"Gaps to address","weaknesses","△"),(c3,"Action plan","recommendations","→")]:
        with col:
            st.markdown(f"#### {icon} {title}")
            items = result[key] or ["No items returned."]
            for item in items:
                st.markdown(f"- {item}")
    st.markdown("#### Scenario fit")
    st.info(result["scenario_fit"])

def render_optimizer(original: str, optimized: str, changes: list[str]) -> None:
    st.markdown("### Prompt optimizer")
    if not optimized:
        st.warning("The AI did not return an optimized prompt. Evaluate again to retry.")
        return
    left,right = st.columns(2)
    with left:
        st.markdown("**Original prompt**")
        st.text_area("Original", original, height=260, disabled=True, label_visibility="collapsed")
    with right:
        st.markdown("**Optimized prompt**")
        st.text_area("Optimized", optimized, height=260, key="optimized_display", label_visibility="collapsed")
    diff = difflib.ndiff(original.split(), optimized.split())
    html_parts = []
    for token in diff:
      safe_token = html.escape(token[2:])

    if token.startswith("+ "):
        html_parts.append(f"<span class='added'>{safe_token}</span>")
    elif token.startswith("- "):
        html_parts.append(f"<span class='removed'>{safe_token}</span>")
    else:
        html_parts.append(safe_token)
    st.markdown("**Change highlighting**")
    st.markdown("<div style='background:#111a2e;border:1px solid #34415f;border-radius:12px;padding:15px;line-height:2'>" + " ".join(html_parts) + "</div>", unsafe_allow_html=True)
    st.markdown("**What changed and why**")
    for item in changes or ["The optimized version clarifies the task and improves instruction structure."]:
        st.markdown(f"- {item}")
    if st.button("↻ Use optimized prompt in workspace", use_container_width=True):
        st.session_state.prompt_text = optimized
        st.session_state.page = "Practice & Evaluate"
        st.rerun()

def render_progress(history: list[dict[str, Any]]) -> None:
    st.markdown('<div class="pf-kicker">LEARNING ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="pf-title">Progress tracker</div>', unsafe_allow_html=True)
    if not history:
        st.info("No completed evaluations yet. Your progress will appear here after your first successful evaluation.")
        return
    render_metric_row(history)
    df = pd.DataFrame(history)
    df["Attempt"] = range(1, len(df)+1)
    st.markdown("#### Overall score trend")
    fig = px.line(df, x="Attempt", y="overall", markers=True, hover_data=["scenario_title","timestamp"])
    fig.update_traces(line_color="#a78bfa", line_width=3, marker_size=9)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=320, yaxis=dict(range=[0,100],gridcolor="#303b57"), xaxis=dict(gridcolor="#303b57"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("#### Dimension trends")
    rows = []
    for attempt_number, item in enumerate(history, start=1):
      for dim in DIMENSIONS:
        value = item.get("scores", {}).get(dim)
        if value is not None:
            rows.append({
                "Attempt": attempt_number,
                "Dimension": dim,
                "Score": float(value)
            })

    trend_df = pd.DataFrame(rows)

    fig2 = px.line(
        trend_df,
        x="Attempt",
        y="Score",
        color="Dimension",
        markers=True,
        category_orders={"Dimension": DIMENSIONS}
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        height=340,
        yaxis=dict(range=[0,10], gridcolor="#303b57"),
        xaxis=dict(gridcolor="#303b57")
    )

    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    table = pd.DataFrame([
        {
            "Date": x["timestamp"],
            "Scenario": x["scenario_title"],
            "Score": x["overall"],
            "Skill": skill_level(x["overall"])
        }
        for x in history
    ])

    st.markdown("#### Evaluation history")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_df = pd.DataFrame([
        {
            "timestamp": x["timestamp"],
            "scenario": x["scenario_title"],
            "prompt": x["prompt"],
            "overall_score": x["overall"],
            **{f"{k.lower()}_score": v for k, v in x["scores"].items()}
        }
        for x in history
    ])

    st.download_button(
        "Download history as CSV",
        csv_df.to_csv(index=False).encode("utf-8"),
        "promptforge_history.csv",
        "text/csv",
        use_container_width=True
    )

    if st.button("Clear evaluation history", type="secondary"):
        save_history([])
        st.rerun()

def main() -> None:
    history = load_history()
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    with st.sidebar:
        st.markdown("# ✦ PromptForge")
        st.caption("Build Better Prompts. Master AI.")
        st.markdown("---")
        pages = ["Dashboard", "Practice & Evaluate", "Prompt Optimizer", "Progress Tracker"]
        selected = st.radio("WORKSPACE", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
        st.session_state.page = selected
        st.markdown("---")
        st.markdown("**AI ENGINE**")
        st.caption(f"Groq · {MODEL_NAME}")
        st.markdown('<span class="pf-pill">LLM-as-a-Judge</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("PromptForge · Mini Project")
    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard(history)
    elif page == "Practice & Evaluate":
        st.markdown('<div class="pf-kicker">PROMPT WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="pf-title">Practice. Evaluate. Improve.</div>', unsafe_allow_html=True)
        st.caption("Choose a real-world challenge or evaluate a prompt without a scenario.")
        domains = ["All domains"] + sorted({s["domain"] for s in SCENARIOS})
        domain = st.selectbox("Filter by domain", domains)
        difficulties = ["All levels", "Beginner", "Intermediate", "Advanced"]
        difficulty = st.selectbox("Filter by difficulty", difficulties)
        filtered = [s for s in SCENARIOS if (domain == "All domains" or s["domain"] == domain) and (difficulty == "All levels" or s["difficulty"] == difficulty)]
        options = ["General evaluation — no scenario"] + [f"{s['title']} · {s['difficulty']} · {s['domain']}" for s in filtered]
        chosen = st.selectbox("Choose a practice scenario", options)
        scenario = None if chosen.startswith("General") else next(s for s in filtered if chosen.startswith(s["title"]))
        if scenario:
            with st.container(border=True):
                st.markdown(f"**{scenario['title']}**  ·  `{scenario['domain']}`  ·  `{scenario['difficulty']}`")
                st.write(scenario["situation"])
                st.markdown("**Scenario requirements**")
                for req in scenario["requirements"]:
                    st.markdown(f"- {req}")
                st.caption(f"Hint: {scenario['hint']}")
        prompt = st.text_area("Your prompt", key="prompt_text", height=230, placeholder="Example: Act as a customer support specialist. Write a helpful response to...")
        words = len(prompt.split())
        chars = len(prompt)
        st.caption(f"{words} words · {chars} characters")
        if st.button("✦ Evaluate my prompt", use_container_width=True):
            if len(prompt.strip()) < 12:
                st.warning("Please enter a more complete prompt before evaluation.")
            else:
                with st.spinner("PromptForge is analyzing your prompt across six dimensions..."):
                    try:
                        result = evaluate_prompt(prompt.strip(), scenario)
                        score = overall_score(result["scores"])
                        record = {"timestamp":datetime.now().astimezone().isoformat(timespec="seconds"),"scenario_id":scenario["id"] if scenario else "GENERAL","scenario_title":scenario["title"] if scenario else "General evaluation","prompt":prompt.strip(),"scores":result["scores"],"overall":score,"justifications":result["justifications"],"strengths":result["strengths"],"weaknesses":result["weaknesses"],"recommendations":result["recommendations"],"scenario_fit":result["scenario_fit"],"optimized_prompt":result["optimized_prompt"],"changes":result["changes"]}
                        save_history(load_history() + [record])
                        st.session_state.latest_result = record
                        st.success("Evaluation completed and saved to your progress history.")
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Could not complete the evaluation. Check your API key, connection and Groq account limits. Details: {exc}")
        if "latest_result" in st.session_state:
            render_feedback(st.session_state.latest_result, st.session_state.latest_result["overall"])
    elif page == "Prompt Optimizer":
        st.markdown('<div class="pf-kicker">REFINEMENT STUDIO</div>', unsafe_allow_html=True)
        st.markdown('<div class="pf-title">Turn good prompts into great ones.</div>', unsafe_allow_html=True)
        latest = st.session_state.get("latest_result")
        if latest:
            render_optimizer(latest["prompt"], latest["optimized_prompt"], latest["changes"])
        else:
            st.info("Evaluate a prompt first. The optimizer will use the AI-generated improved version and explain its changes.")
            st.button("Go to Practice & Evaluate", on_click=lambda: setattr(st.session_state, "page", "Practice & Evaluate"))
    elif page == "Progress Tracker":
        render_progress(history)

if __name__ == "__main__":
    main()
