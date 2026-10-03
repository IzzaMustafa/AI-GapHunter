"""GapHunter AI agent module."""
from __future__ import annotations

import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task, LLM

MODEL = "groq/openai/gpt-oss-120b"


def _llm(api_key: str) -> LLM:
    return LLM(model=MODEL, api_key=api_key, temperature=0.25)


def _parse_json(raw: Any) -> dict:
    text = getattr(raw, "raw", None) or str(raw)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"raw_output": text, "parse_warning": "Model output was not valid JSON."}


def _run(role: str, goal: str, backstory: str, description: str,
         expected_output: str, api_key: str) -> dict:
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=_llm(api_key),
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return _parse_json(crew.kickoff())

def run_competitor_agent(problems: dict, market: dict, rag_context: list[dict], api_key: str) -> dict:
    return _run(
        "Competitor and Existing Solution Analyst",
        "Identify only supported existing solutions and assess how fully problems appear solved.",
        "You never invent competitors, pricing, market facts or product features.",
        f"""Problems:
{json.dumps(problems, ensure_ascii=False)}
Market analysis:
{json.dumps(market, ensure_ascii=False)}
Retrieved competitor/startup evidence:
{json.dumps(rag_context, ensure_ascii=False)}

Return ONLY JSON:
{{"analyses":[{{"problem":"", "existing_solutions":[],
"solution_status":"Unknown", "strengths":[], "weaknesses":[],
"pricing_or_business_model":[], "target_users":[], "evidence":[],
"unknowns":[]}}]}}
solution_status must be one of: Already solved, Partially solved, Poorly solved,
Underserved, Potential gap, Unknown.
Do not name a competitor unless supported by retrieved evidence.""",
        "Valid JSON with competitor analyses.",
        api_key,
    )
