# GapHunter AI

### Turn your skills and interests into testable business opportunities

GapHunter AI is a Streamlit application that helps students, aspiring founders, and builders explore opportunities suited to their skills, interests, budget, and resources.

Built for a final hackathon, the project combines a ten-agent AI workflow with a local research knowledge base to move from problem discovery to practical validation plans.

## What it does

- Collects your skills, interests, experience, resources, and constraints.
- Finds relevant problems and researches market signals using local documents.
- Examines existing solutions and identifies potential gaps.
- Develops opportunity candidates and challenges their assumptions.
- Assesses feasibility and explains qualitative trade-offs.
- Generates reports with MVP suggestions, risks, and validation experiments.

## The ten-agent workflow

1. Profile Analyst
2. Problem Hunter
3. Market and Trend Researcher
4. Competitor and Existing Solution Analyst
5. Gap Finder
6. Innovation Strategist
7. Red Team
8. Feasibility Analyst
9. Opportunity Evaluator
10. Opportunity Refiner

## Features

- Custom dark interface with opportunity cards and detailed tabs
- Live agent completion and error indicators
- Local document retrieval using embeddings and FAISS
- Evidence, inference, and hypothesis sections
- On-demand challenge and improvement of selected opportunities
- Fictional sample preview without AI calls
- Full analysis download as JSON
- Token budgeting for selected agents and bounded quota retries

## Tools and technologies

| Component | Technology |
|-----------|------------|
| Application | Python |
| Interface | Streamlit and custom CSS |
| Agent orchestration | CrewAI |
| Language model | OpenAI GPT OSS 120B served through Groq |
| Model gateway | LiteLLM |
| Embeddings | SentenceTransformers with all-MiniLM-L6-v2 |
| Vector retrieval | FAISS |
| Document extraction | pypdf |
| Token counting | tiktoken |
| Vector processing | NumPy |
| Version control | GitHub |
| Deployment | Streamlit Community Cloud |

## How research retrieval works

Documents in `knowledge_base/` are extracted, divided into overlapping chunks, and converted into embeddings. FAISS retrieves relevant passages for the research stages.

The index is saved and reused when its documents and configuration remain compatible. Changes to the knowledge base trigger rebuilding.

**The current application uses local documents. Live web search is a proposed extension.**

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/IzzaMustafa/AI-GapHunter.git
cd AI-GapHunter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Keep this file out of GitHub. Add these entries to `.gitignore`:

```gitignore
.streamlit/secrets.toml
.venv/
__pycache__/
knowledge_base/.faiss/
```

For Streamlit Community Cloud, configure the key through the app's Secrets settings.

### 5. Launch the application

```bash
streamlit run app.py
```

The first research run may take longer while the embedding model downloads and the document index is created.

## Project structure

```text
AI-GapHunter/
├── .streamlit/
│   └── config.toml
├── knowledge_base/
├── app.py
├── theme.css
├── prompt_budget.py
├── profile_agent.py
├── problem_agent.py
├── market_agent.py
├── competitor_agent.py
├── gap_agent.py
├── innovation_agent.py
├── red_team_agent.py
├── feasibility_agent.py
├── evaluator_agent.py
├── final_agent.py
└── requirements.txt
```

## Limitations

- Suggestions are starting points for research, not validated business opportunities.
- Research quality depends on the coverage and freshness of the local documents.
- AI-generated claims and source references require human review.
- Groq quotas can interrupt stages. A displayed report may be incomplete if warnings appear.
- Token budgeting may shorten supporting detail in large inputs.
- The MVP has no user accounts or durable analysis history.

## Future improvements

- Optional live search with source links
- Consistent token and quota handling across all agents
- PDF and Word report exports
- Saved analyses and user feedback
- Broader research coverage and multilingual testing

## Author

**Izza Mustafa Jadoon**

BS Artificial Intelligence

---

## License

This project is developed for educational and research purposes.
