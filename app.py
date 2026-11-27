"""
PromptForge — AI-Powered Prompt Evaluation Studio
Run with: streamlit run app.py
"""
from __future__ import annotations

import csv
import difflib
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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 85% 0%, rgba(103,80,220,.16), transparent 28%), #0b1020; color:#edf0ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#11182b,#0d1425); border-right:1px solid rgba(151,139,255,.16);}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em;}
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1500px;}
[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(31,42,69,.94),rgba(19,28,49,.94)); padding:18px 20px; border:1px solid rgba(151,139,255,.18); border-radius:17px;}
[data-testid="stMetricLabel"] {color:#aab5d1;}
[data-testid="stMetricValue"] {color:#f4f5ff; font-family:'Space Grotesk',sans-serif;}
div.stButton > button, div.stDownloadButton > button {border-radius:11px; border:1px solid rgba(157,143,255,.35); background:linear-gradient(110deg,#6754d9,#8b5cf6); color:white; font-weight:700; min-height:2.7rem;}
div.stButton > button:hover, div.stDownloadButton > button:hover {border-color:#c4b5fd; color:white; filter:brightness(1.12);}
div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {background:#111a2e; color:#f3f4ff; border:1px solid #34415f; border-radius:12px;}
div[data-testid="stSelectbox"] > div > div {background:#111a2e;}
.pf-card {background:linear-gradient(145deg,rgba(29,40,67,.96),rgba(17,26,46,.96)); border:1px solid rgba(151,139,255,.18); padding:19px; border-radius:17px; min-height:112px;}
.pf-muted {color:#9eabc8; font-size:.88rem;}
.pf-kicker {color:#a99aff; font-weight:700; letter-spacing:.12em; font-size:.72rem; text-transform:uppercase;}
.pf-title {font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#f5f4ff;}
.pf-pill {display:inline-block;padding:5px 10px;border-radius:30px;background:rgba(139,92,246,.15);color:#c4b5fd;border:1px solid rgba(167,139,250,.25);font-size:.75rem;}
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
        raise ValueError("The AI response had an unexpected structure.")
    scores = obj.get("scores")
    justifications = obj.get("justifications")
    if not isinstance(scores, dict) or not isinstance(justifications, dict):
        raise ValueError("The AI response is missing scores or justifications.")
    normalized_scores = {}
    normalized_justifications = {}
    for dim in DIMENSIONS:
        try:
            value = float(scores[dim])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"The AI response is missing a valid {dim} score.")
        normalized_scores[dim] = max(0.0, min(10.0, value))
        normalized_justifications[dim] = str(justifications.get(dim, "No justification returned."))
    def clean_list(key: str) -> list[str]:
        value = obj.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(x).strip() for x in value if str(x).strip()][:3]
    return {
        "scores": normalized_scores,
        "justifications": normalized_justifications,
        "strengths": clean_list("strengths"),
        "weaknesses": clean_list("weaknesses"),
        "recommendations": clean_list("recommendations"),
        "scenario_fit": str(obj.get("scenario_fit", "No scenario-fit comment returned.")),
        "optimized_prompt": str(obj.get("optimized_prompt", "")).strip(),
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
    st.markdown('<div class="pf-kicker">PROMPT INTELLIGENCE / OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="pf-title">Your prompt command center</div>', unsafe_allow_html=True)
    st.caption("A live view of your evaluated attempts. Metrics appear after real Groq evaluations.")
    render_metric_row(history)
    if not history:
        st.info("Your dashboard is ready. Go to **Practice & Evaluate** to submit your first prompt and generate real evaluation data.")
        return
    latest = history[-1]
    left,right = st.columns([1,1])
    with left:
        st.markdown("#### Latest prompt quality")
        st.plotly_chart(make_gauge(latest["overall"]), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<span class="pf-pill">{skill_level(latest["overall"])} level</span>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Six-dimension profile")
        st.plotly_chart(make_radar(latest["scores"]), use_container_width=True, config={"displayModeBar":False})
    st.markdown("#### Component performance")
    frame = pd.DataFrame({"Dimension": DIMENSIONS, "Score": [latest["scores"][d] for d in DIMENSIONS]})
    fig = px.bar(frame, x="Score", y="Dimension", orientation="h", range_x=[0,10], text="Score")
    fig.update_traces(marker_color=["#ef6464" if x < 5 else "#e9ad4e" if x <= 7 else "#39c995" for x in frame["Score"]], textposition="outside")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=300, margin=dict(l=10,r=30,t=10,b=10), xaxis=dict(gridcolor="#303b57"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    if len(history) > 1:
        st.markdown("#### Your improvement journey")
        trend = pd.DataFrame([{"Attempt":i+1,"Score":x["overall"],"Date":x["timestamp"][:10]} for i,x in enumerate(history)])
        fig2 = px.line(trend, x="Attempt", y="Score", markers=True, hover_data=["Date"])
        fig2.update_traces(line_color="#a78bfa", line_width=3, marker_size=9)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=270, yaxis=dict(range=[0,100],gridcolor="#303b57"), xaxis=dict(gridcolor="#303b57"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

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
        if token.startswith("+ "):
            html_parts.append(f"<span style='background:#14532d;color:#bbf7d0;padding:2px 4px;border-radius:4px'>{token[2:]}</span>")
        elif token.startswith("- "):
            html_parts.append(f"<span style='background:#7f1d1d;color:#fecaca;text-decoration:line-through;padding:2px 4px;border-radius:4px'>{token[2:]}</span>")
        elif token.startswith("  "):
            html_parts.append(token[2:])
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
    for item in history:
        for dim, val in item["scores"].items():
            rows.append({"Attempt":item["timestamp"][:10], "Dimension":dim, "Score":val})
    trend_df = pd.DataFrame(rows)
    fig2 = px.line(trend_df, x="Attempt", y="Score", color="Dimension", markers=True)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", height=340, yaxis=dict(range=[0,10],gridcolor="#303b57"), xaxis=dict(gridcolor="#303b57"))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
    table = pd.DataFrame([{"Date":x["timestamp"],"Scenario":x["scenario_title"],"Score":x["overall"],"Skill":skill_level(x["overall"])} for x in history])
    st.markdown("#### Evaluation history")
    st.dataframe(table, use_container_width=True, hide_index=True)
    csv_df = pd.DataFrame([{"timestamp":x["timestamp"],"scenario":x["scenario_title"],"prompt":x["prompt"],"overall_score":x["overall"],**{f"{k.lower()}_score":v for k,v in x["scores"].items()}} for x in history])
    st.download_button("Download history as CSV", csv_df.to_csv(index=False).encode("utf-8"), "promptforge_history.csv", "text/csv", use_container_width=True)
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
