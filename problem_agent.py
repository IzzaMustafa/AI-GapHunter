"""GapHunter AI agent module."""
from __future__ import annotations

import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task, LLM

from prompt_budget import BudgetedLLM, CONTEXT_NOTE, compact_contexts, validate_task

MODEL = "groq/openai/gpt-oss-120b"


def _llm(api_key: str) -> LLM:
    return BudgetedLLM(model=MODEL, api_key=api_key, temperature=0.25,
                       max_tokens=2000, reasoning_effort="low",
                       timeout=90, max_retries=0, num_retries=0)


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
    validate_task(description)
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=_llm(api_key),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
    )
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return _parse_json(crew.kickoff())

def run_problem_agent(profile_analysis: dict, profile: dict, rag_context: list[dict], api_key: str) -> dict:
    context = compact_contexts({"profile": profile, "analysis": profile_analysis,
                               "evidence": rag_context})
    return _run(
        "Problem Hunter",
        "Find specific, recurring and potentially painful real-world problems that fit the user.",
        "You avoid vague problem statements and separate evidence from hypotheses.",
        f"""{CONTEXT_NOTE}
User profile:
{context['profile']}
Profile analysis:
{context['analysis']}
Retrieved knowledge-base evidence:
{context['evidence']}

Generate 4 to 6 candidate problems. Return ONLY JSON:
{{"problems":[{{"problem":"", "affected_users":[], "pain_or_inefficiency":"",
"why_relevant_to_user":"", "evidence":[], "inferences":[], "hypotheses":[]}}]}}
Do not fabricate facts or sources. Evidence may only cite the retrieved context.""",
        "Valid JSON containing a problems array.",
        api_key,
    )
