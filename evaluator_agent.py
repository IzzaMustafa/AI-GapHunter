"""GapHunter AI agent module."""
from __future__ import annotations

import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task, LLM
from prompt_budget import BudgetedLLM, CONTEXT_NOTE, compact_contexts, validate_task, run_with_rate_limit_retry

MODEL = "groq/openai/gpt-oss-120b"


def _llm(api_key: str) -> LLM:
    return BudgetedLLM(model=MODEL, api_key=api_key, temperature=0.25,
               max_tokens=2000, reasoning_effort="low")


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
    def execute():
        agent = Agent(role=role, goal=goal, backstory=backstory, llm=_llm(api_key),
                      verbose=False, allow_delegation=False, max_iter=1,
                      max_retry_limit=0)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return crew.kickoff()
    return _parse_json(run_with_rate_limit_retry(execute))

def run_evaluator_agent(opportunities: dict, gaps: dict, competitors: dict,
                        red_team: dict, feasibility: dict, api_key: str) -> dict:
    context = compact_contexts({"opportunities": opportunities, "gaps": gaps,
                               "competitors": competitors, "red_team": red_team,
                               "feasibility": feasibility})
    return _run(
        "Opportunity Evaluator",
        "Explain trade-offs across opportunities using transparent qualitative criteria.",
        "Your labels summarize analysis, not objective startup success predictions.",
        f"""{CONTEXT_NOTE}
Opportunities:
{context['opportunities']}
Gaps:
{context['gaps']}
Competitors:
{context['competitors']}
Red team:
{context['red_team']}
Feasibility:
{context['feasibility']}

Return ONLY JSON:
{{"evaluations":[{{"opportunity_name":"",
"problem_clarity":"Low|Medium|Strong", "user_relevance":"Low|Medium|Strong",
"differentiation":"Low|Medium|Strong", "evidence_strength":"Low|Medium|Strong",
"competition_intensity":"Low|Medium|High", "feasibility":"Low|Medium|High",
"mvp_simplicity":"Low|Medium|High", "resource_compatibility":"Low|Medium|High",
"potential_value":"Low|Medium|Strong", "strong_points":[], "concerns":[],
"validation_priorities":[]}}]}}
Do not rank winners or predict success.""",
        "Valid JSON containing qualitative evaluations and trade-offs.",
        api_key,
    )
