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
               max_tokens=2400, reasoning_effort="low")


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

def run_final_agent(profile: dict, profile_analysis: dict, opportunities: dict, gaps: dict,
                    competitors: dict, market: dict, red_team: dict, feasibility: dict,
                    evaluation: dict, api_key: str) -> dict:
    context = compact_contexts({"profile": profile, "profile_analysis": profile_analysis,
                               "opportunities": opportunities, "gaps": gaps,
                               "competitors": competitors, "market": market,
                               "red_team": red_team, "feasibility": feasibility,
                               "evaluation": evaluation})
    return _run(
        "Opportunity Refiner and Final Strategist",
        "Synthesize the investigation into polished, evidence-aware opportunity reports.",
        "You preserve uncertainty, separate evidence from inference and propose cheap validation before heavy investment.",
        f"""{CONTEXT_NOTE}
User profile:
{context['profile']}
Profile analysis:
{context['profile_analysis']}
Opportunities:
{context['opportunities']}
Gaps:
{context['gaps']}
Competitors:
{context['competitors']}
Market:
{context['market']}
Red team:
{context['red_team']}
Feasibility:
{context['feasibility']}
Evaluation:
{context['evaluation']}

Return ONLY JSON:
{{"opportunities":[{{"name":"", "problem":"", "who_has_problem":[],
"proposed_solution":"", "why_now":"", "existing_solutions":[],
"observed_gap":"", "unique_angle":"", "target_users":[],
"why_you":"", "business_model":[], "mvp":"", "feasibility_summary":"",
"major_risks":[], "evidence":[], "inferences":[], "hypotheses":[],
"validation_experiments":[{{"assumption":"", "test":"", "success_signal_heuristic":""}}],
"what_needs_validation":[]}}], "overall_research_limitations":[]}}
Any suggested threshold must be clearly described as a heuristic, not a fact.
Keep each string to one short sentence and include one validation experiment per opportunity.""",
        "Valid JSON containing final opportunity reports.",
        api_key,
    )
