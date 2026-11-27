# PromptForge

**Build Better Prompts. Master AI.**

PromptForge is an AI-powered prompt evaluation studio built with Streamlit, Groq and Plotly. It evaluates submitted prompts across six dimensions: Role, Goal, Context, Constraints, Format and Style.

## Features

- 15 built-in scenarios across Customer Support, Marketing, HR, Data & Software, and Finance.
- Beginner, Intermediate and Advanced scenario levels.
- General prompt evaluation without selecting a scenario.
- Groq-powered LLM-as-a-Judge evaluation using `openai/gpt-oss-20b`.
- Evidence-based justifications, strengths, weaknesses and actionable recommendations.
- Weighted overall score calculated in Python.
- Interactive dashboard with score gauge, radar chart, component chart and improvement trend.
- AI prompt optimizer with original/optimized comparison and highlighted changes.
- Progress history persisted to `history.json` at runtime.
- CSV export and clear-history control.

## Project files

- `app.py` — complete Streamlit application.
- `.env` — Groq API key configuration (placeholder only).
- `requirements.txt` — Python dependencies.
- `README.md` — setup and usage guide.

## Requirements

- Python 3.10 or newer
- Internet connection for Groq API calls
- A Groq API key

## Setup

1. Extract `PromptForge.zip`.
2. Open the extracted `PromptForge` folder in VS Code.
3. Open the VS Code terminal in that folder.
4. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configure Groq

1. Visit https://console.groq.com/ and sign in.
2. Create an API key in the Groq Console.
3. Open `.env` in the project folder.
4. Replace `your_groq_api_key_here` with your own key:

   ```env
   GROQ_API_KEY=your_actual_groq_api_key
   ```

5. Save `.env`. Do not share your real API key or commit it to a public repository.

The application reads the key from `.env` using `python-dotenv`. If the key is missing or is still the placeholder, the app displays a helpful message.

## Run

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501). Open it in your browser.

## How to use

1. Open **Practice & Evaluate**.
2. Choose a domain, difficulty and scenario, or select general evaluation.
3. Read the scenario requirements if a scenario is selected.
4. Enter a prompt and select **Evaluate my prompt**.
5. Review the score, six dimension scores, evidence, strengths, weaknesses and recommendations.
6. Open **Prompt Optimizer** to review the improved prompt and highlighted changes.
7. Re-evaluate the optimized prompt to compare attempts.
8. Open **Progress Tracker** to review score trends, export CSV history or clear history.

## Scoring

The judge returns a score from 0 to 10 for each dimension. The application computes the overall score using the assignment weights:

- Goal: 20
- Context: 20
- Constraints: 15
- Format: 15
- Role: 10
- Style: 10

The weights total 90. The normalized overall score is:

`overall = (sum(dimension_score * weight) / 90) * 10`

This produces a score out of 100. The arithmetic is performed in Python, not delegated to the model.

## Troubleshooting

- **Missing API key:** Check that `.env` is in the same folder as `app.py`, and that the variable is named `GROQ_API_KEY`.
- **Authentication error:** Create a fresh key in Groq Console and update `.env`.
- **Rate limit:** Wait and retry after your Groq account's limit resets.
- **Network error:** Check your internet connection.
- **Model unavailable:** Confirm the model name `openai/gpt-oss-20b` is available to your Groq account.
- **Invalid AI response:** PromptForge validates the JSON response and retries once.

## Notes

- Real evaluation requires a valid Groq API key and internet access.
- `history.json` is created automatically after the first successful evaluation. It is runtime data and is not included in the initial ZIP.
- The ZIP contains the four requested deliverable files at its root.
🚀 Live Demo

Try my website here: "Open PromptForge" - https://promtforge-tanuja337.streamlit.app/

💻 Technologies Used

- Python
- Streamlit
