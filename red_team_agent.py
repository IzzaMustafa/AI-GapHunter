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

def run_red_team_agent(opportunities: dict, competitors: dict, market: dict, api_key: str) -> dict:
    return _run(
        "Red Team",
        "Aggressively challenge startup opportunities and expose what must be validated.",
        "You are skeptical and independent. You do not agree merely because another agent proposed an idea.",
        f"""Opportunities:
{json.dumps(opportunities, ensure_ascii=False)}
Competitor analysis:
{json.dumps(competitors, ensure_ascii=False)}
Market analysis:
{json.dumps(market, ensure_ascii=False)}

Return ONLY JSON:
{{"attacks":[{{"opportunity_name":"", "why_it_might_fail":[],
"weak_assumptions":[], "competitor_threats":[], "market_risks":[],
"technical_risks":[], "customer_acquisition_risks":[],
"what_must_be_validated":[], "severity":"Low|Medium|High"}}]}}
Do not invent competitor facts.""",
        "Valid JSON containing skeptical red-team attacks.",
        api_key,
    )
