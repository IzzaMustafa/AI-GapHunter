"""GapHunter AI agent module."""
from __future__ import annotations

import json
import re
import time
import math
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
    from litellm import RateLimitError
    validate_task(description)
    for attempt in range(3):
        # A fresh Crew prevents failed attempts from adding conversation history.
        agent = Agent(role=role, goal=goal, backstory=backstory,
                      llm=_llm(api_key), verbose=False, allow_delegation=False,
                      max_iter=1, max_retry_limit=0)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        try:
            return _parse_json(crew.kickoff())
        except RateLimitError as exc:
            message = str(exc).lower()
            # Oversized requests and daily quotas will not benefit from a short wait.
            if (attempt == 2 or "request too large" in message
                    or "tokens per day" in message or "(tpd)" in message
                    or not ("tokens per minute" in message or "(tpm)" in message)):
                raise
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            try:
                delay = float(headers.get("retry-after", "nan"))
            except (TypeError, ValueError):
                delay = float("nan")
            if not math.isfinite(delay):
                match = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", message)
                delay = (60 * float(match.group(1) or 0) + float(match.group(2))) if match else 61
            if delay > 120:
                raise
            delay = max(delay, 1) + 0.5
            print(f"[GapHunter] Feasibility minute quota: retry {attempt + 1}/2 in {delay:.1f}s", flush=True)
            while delay > 0:
                interval = min(delay, 30)
                time.sleep(interval)
                delay -= interval


def run_feasibility_agent(opportunities: dict, profile: dict, red_team: dict, api_key: str) -> dict:
    context = compact_contexts({"opportunities": opportunities, "profile": profile,
                               "red_team": red_team})
    return _run(
        "Feasibility Analyst",
        "Assess whether the user can realistically prototype each opportunity.",
        "You focus on buildability, resources, data, infrastructure, cost drivers and MVP scope.",
        f"""{CONTEXT_NOTE}
Opportunities:
{context['opportunities']}
User profile/resources:
{context['profile']}
Red-team findings:
{context['red_team']}

Return ONLY JSON:
{{"feasibility":[{{"opportunity_name":"", "technical_feasibility":"Low|Medium|High",
"mvp_feasibility":"Low|Medium|High", "estimated_complexity":"Low|Medium|High",
"hardware_requirements":[], "required_skills":[], "infrastructure":[],
"data_requirements":[], "api_cost_considerations":[], "deployment_considerations":[],
"resource_fit":"Low|Medium|High", "key_constraints":[]}}]}}
Never assign startup success probabilities.""",
        "Valid JSON containing qualitative feasibility analysis.",
        api_key,
    )
