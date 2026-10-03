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

def run_profile_agent(profile: dict, api_key: str) -> dict:
    return _run(
        "Profile Analyst",
        "Turn the user's background into a concise opportunity profile.",
        "You map skills, interests, constraints and resources to realistic capability areas. "
        "Never claim startup success and never invent facts about the user.",
        f"""Analyze this user profile:
{json.dumps(profile, ensure_ascii=False)}

Return ONLY JSON with:
strongest_skills, technical_capabilities, interests, constraints, resources,
suitable_domains, skill_interest_combinations, opportunity_profile_summary.
Keep lists concise and grounded only in the supplied profile.""",
        "A valid JSON object containing the requested opportunity profile fields.",
        api_key,
    )
