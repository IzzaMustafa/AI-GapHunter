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

def run_final_agent(profile: dict, profile_analysis: dict, opportunities: dict, gaps: dict,
                    competitors: dict, market: dict, red_team: dict, feasibility: dict,
                    evaluation: dict, api_key: str) -> dict:
    return _run(
        "Opportunity Refiner and Final Strategist",
        "Synthesize the investigation into polished, evidence-aware opportunity reports.",
        "You preserve uncertainty, separate evidence from inference and propose cheap validation before heavy investment.",
        f"""User profile:
{json.dumps(profile, ensure_ascii=False)}
Profile analysis:
{json.dumps(profile_analysis, ensure_ascii=False)}
Opportunities:
{json.dumps(opportunities, ensure_ascii=False)}
Gaps:
{json.dumps(gaps, ensure_ascii=False)}
Competitors:
{json.dumps(competitors, ensure_ascii=False)}
Market:
{json.dumps(market, ensure_ascii=False)}
Red team:
{json.dumps(red_team, ensure_ascii=False)}
Feasibility:
{json.dumps(feasibility, ensure_ascii=False)}
Evaluation:
{json.dumps(evaluation, ensure_ascii=False)}

Return ONLY JSON:
{{"opportunities":[{{"name":"", "problem":"", "who_has_problem":[],
"proposed_solution":"", "why_now":"", "existing_solutions":[],
"observed_gap":"", "unique_angle":"", "target_users":[],
"why_you":"", "business_model":[], "mvp":"", "feasibility_summary":"",
"major_risks":[], "evidence":[], "inferences":[], "hypotheses":[],
"validation_experiments":[{{"assumption":"", "test":"", "success_signal_heuristic":""}}],
"what_needs_validation":[]}}], "overall_research_limitations":[]}}
Any suggested threshold must be clearly described as a heuristic, not a fact.""",
        "Valid JSON containing final opportunity reports.",
        api_key,
    )
