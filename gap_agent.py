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

def run_gap_agent(profile_analysis: dict, problems: dict, market: dict, competitors: dict,
                  rag_context: list[dict], api_key: str) -> dict:
    return _run(
        "Gap Finder",
        "Convert supported problem and solution weaknesses into testable opportunity gaps.",
        "You explain why a gap may exist and clearly label uncertainty.",
        f"""Profile:
{json.dumps(profile_analysis, ensure_ascii=False)}
Problems:
{json.dumps(problems, ensure_ascii=False)}
Market:
{json.dumps(market, ensure_ascii=False)}
Competitors:
{json.dumps(competitors, ensure_ascii=False)}
Relevant retrieved evidence:
{json.dumps(rag_context, ensure_ascii=False)}

Return ONLY JSON:
{{"gaps":[{{"gap_name":"", "problem":"", "gap_types":[],
"underserved_users":[], "why_gap_may_exist":"", "evidence":[],
"inferences":[], "hypotheses":[], "initial_opportunity":""}}]}}
Generate up to 4 strong, specific gaps. Do not claim uniqueness.""",
        "Valid JSON containing evidence-aware opportunity gaps.",
        api_key,
    )
