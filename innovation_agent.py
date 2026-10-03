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

def run_innovation_agent(gaps: dict, profile_analysis: dict, api_key: str,
                         red_team_feedback: dict | None = None) -> dict:
    return _run(
        "Innovation Strategist",
        "Create meaningfully differentiated angles for supported gaps.",
        "You explore audience, workflow, technology, localization, pricing and AI-native differentiation.",
        f"""Gaps:
{json.dumps(gaps, ensure_ascii=False)}
User capability profile:
{json.dumps(profile_analysis, ensure_ascii=False)}
Red-team feedback, if this is a refinement cycle:
{json.dumps(red_team_feedback or {}, ensure_ascii=False)}

Return ONLY JSON:
{{"opportunities":[{{"name":"", "problem":"", "target_users":[],
"proposed_solution":"", "unique_angles":[], "why_now_hypothesis":"",
"business_models":[], "mvp":"", "differentiation_hypothesis":"",
"assumptions":[]}}]}}
Return up to 4 opportunities. Do not claim they have never been done before.""",
        "Valid JSON containing differentiated opportunity candidates.",
        api_key,
    )
