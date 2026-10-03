from __future__ import annotations

import json
import html
import time
from pathlib import Path
import streamlit as st

# CrewAI 1.14/1.15 may add an Anthropic-only ``cache_breakpoint``
# field to agent messages. Groq rejects that unsupported field. Disable
# CrewAI's marker before importing/creating any agents.
try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda message: message
except (ImportError, AttributeError):
    # Newer CrewAI versions may already handle provider compatibility.
    pass

from profile_agent import run_profile_agent
from problem_agent import run_problem_agent
from market_agent import retrieve_context, run_market_agent
from competitor_agent import run_competitor_agent
from gap_agent import run_gap_agent
from innovation_agent import run_innovation_agent
from red_team_agent import run_red_team_agent
from feasibility_agent import run_feasibility_agent
from evaluator_agent import run_evaluator_agent
from final_agent import run_final_agent


st.set_page_config(page_title="GapHunter AI | Opportunity Studio", page_icon="✦", layout="wide")
theme_path = Path(__file__).with_name("theme.css")
if not theme_path.is_file():
    st.error("theme.css is missing from this deployment. Upload it beside app.py on the branch deployed by Streamlit, commit the change, and reboot the app.")
    st.stop()
st.markdown("<style>" + theme_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

STAGES = [
    ("profile", "Profile", "Understanding your profile", "Maps your skills, interests and constraints."),
    ("problems", "Problems", "Hunting for problems", "Identifies problems worth investigating."),
    ("market", "Market", "Researching market trends", "Checks demand signals and local evidence."),
    ("competitors", "Competitors", "Investigating existing solutions", "Examines existing products and alternatives."),
    ("gaps", "Gaps", "Searching for gaps", "Looks for underserved needs and workflows."),
    ("innovation", "Innovation", "Exploring unique angles", "Develops differentiated opportunity candidates."),
    ("red_team", "Red team", "Red-teaming opportunities", "Challenges assumptions and failure risks."),
    ("feasibility", "Feasibility", "Checking feasibility", "Checks resources, buildability and MVP scope."),
    ("evaluation", "Trade-offs", "Evaluating trade-offs", "Compares opportunities using qualitative criteria."),
    ("final", "Refinement", "Refining opportunities", "Synthesizes findings and validation experiments."),
]
STAGE_KEYS = {label: key for key, _, label, _ in STAGES}
activity_slot = None


def esc(value):
    return html.escape(str(value), quote=True)


def text_value(value):
    if isinstance(value, dict):
        return "; ".join(f"{str(k).replace('_', ' ')}: {text_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(text_value(item) for item in value)
    return str(value) if value is not None else "Not available"


def short(value, limit=230):
    text = text_value(value)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def section(title, subtitle=""):
    st.markdown(f'<div class="gh-heading"><h2>{esc(title)}</h2>'
                f'<span class="gh-muted">{esc(subtitle)}</span></div>', unsafe_allow_html=True)


def activity_html(activity):
    cards = []
    labels = {"queued": "Queued", "running": "Working", "done": "Complete", "error": "Needs attention"}
    for i, (key, title, _, purpose) in enumerate(STAGES):
        state = activity.get(key, {}).get("state", "queued")
        if state not in labels:
            state = "queued"
        cards.append(f'<div class="gh-agent {state}"><div class="gh-agent-top">'
                     f'<span class="gh-dot"></span>{i+1:02d} · {labels[state]}</div>'
                     f'<strong>{esc(title)}</strong><p>{esc(purpose)}</p></div>')
    return '<div class="gh-agents">' + ''.join(cards) + '</div>'


def render_activity():
    if activity_slot is None:
        return
    activity = st.session_state.get("agent_activity", {})
    if not activity:
        return
    finished = sum(item.get("state") in {"done", "error"} for item in activity.values())
    warnings = sum(item.get("state") == "error" for item in activity.values())
    current = next((title for key, title, _, _ in STAGES if activity.get(key, {}).get("state") == "running"), None)
    status = f"{current} agent is working" if current else (
        f"Finished with {warnings} stage warning(s)" if finished == 10 and warnings else
        "Your opportunity report is ready" if finished == 10 else "Preparing the next stage and local evidence")
    with activity_slot.container():
        section("Inside your investigation", "Live agent activity")
        st.caption("Follow each agent's role and completion state. Local evidence is prepared between stages.")
        st.progress(finished / 10, text=f"{finished} / 10 stages settled · {status}")
        st.markdown(activity_html(activity), unsafe_allow_html=True)


def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def safe_stage(label, fn, *args):
    key = STAGE_KEYS.get(label)
    start = time.monotonic()
    print(f"[GapHunter] Starting stage: {label}", flush=True)
    if key:
        st.session_state.agent_activity[key] = {"state": "running"}
        render_activity()
    try:
        with st.status(label, expanded=False) as status:
            if key:
                st.caption(next(purpose for stage_key, _, _, purpose in STAGES if stage_key == key))
            result = fn(*args)
            status.update(label=f"✓ {label}", state="complete")
        if key:
            st.session_state.agent_activity[key] = {"state": "done", "seconds": round(time.monotonic() - start, 1)}
            render_activity()
        print(f"[GapHunter] Completed stage: {label} · {time.monotonic() - start:.1f}s", flush=True)
        return result
    except Exception as exc:
        print(f"[GapHunter] Failed stage: {label} · {type(exc).__name__}", flush=True)
        if key:
            st.session_state.agent_activity[key] = {"state": "error", "seconds": round(time.monotonic() - start, 1)}
            render_activity()
        st.warning(f"⚠️ {label} could not be completed. Remaining analysis will use available information.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: stage failed"}


def show_list(title, value):
    if isinstance(value, list) and value:
        body = '<ul>' + ''.join(f'<li>{esc(text_value(item))}</li>' for item in value) + '</ul>'
    else:
        body = f'<p>{esc(text_value(value) if value else "Not available")}</p>'
    st.markdown(f'<div class="gh-field"><h4>{esc(title)}</h4>{body}</div>', unsafe_allow_html=True)


def demo_analysis():
    # Fictional examples for exploring the design. Never presented as real research.
    examples = [
        ("CampusFlow", "University teams coordinate student requests through scattered messages and spreadsheets.",
         "A shared request inbox with routing, ownership and clear status updates.", "One department, three request types and a simple staff dashboard.", "Students and university staff"),
        ("SkillBridge", "Students struggle to connect what they learn with portfolio projects that employers can inspect.",
         "Guided project briefs with checkpoints, practical feedback and portfolio evidence.", "One learning track, five briefs and a small mentor pilot.", "University students"),
        ("LocalLens", "Small businesses have limited time to turn customer feedback into specific service improvements.",
         "A lightweight feedback inbox that groups recurring problems and suggests experiments.", "A feedback form and weekly issue summary for three pilot businesses.", "Small business owners"),
    ]
    opportunities, evaluations = [], []
    for name, problem, solution, mvp, users in examples:
        opportunities.append({"name": name, "problem": problem, "who_has_problem": [users],
            "target_users": [users], "proposed_solution": solution,
            "observed_gap": "A focused workflow with visible ownership and affordable local onboarding.",
            "unique_angle": "Start with one narrow workflow and learn directly from pilot users.",
            "why_you": "Uses Python and web-development skills in a manageable software pilot.",
            "business_model": ["Test a small subscription after validating repeat use."], "mvp": mvp,
            "major_risks": ["Users may already have an adequate workaround.", "Willingness to pay has not been established."],
            "evidence": [], "inferences": ["The proposed workflow could reduce coordination overhead."],
            "hypotheses": ["Pilot users will return to the workflow each week."],
            "validation_experiments": [{"assumption": "The problem occurs often enough to justify a new workflow.",
                "test": "Interview five target users and observe one real workflow.",
                "success_signal_heuristic": "Several users describe recurring friction and agree to try a prototype; this is a heuristic."}],
            "what_needs_validation": ["Frequency of the problem", "Existing alternatives", "Willingness to pay"]})
        evaluations.append({"opportunity_name": name, "problem_clarity": "Medium", "differentiation": "Medium",
                            "feasibility": "High", "evidence_strength": "Low"})
    return {"profile": {}, "problems": {}, "market": {}, "competitors": {}, "gaps": {}, "innovation": {},
            "red_team": {}, "feasibility": {}, "evaluation": {"evaluations": evaluations},
            "final": {"opportunities": opportunities, "overall_research_limitations": ["Fictional design preview. No agent calls, interviews or market research were performed."]}}



def research_context(query, top_k, categories):
    with st.status("Preparing local research before the next agent", expanded=True) as status:
        st.caption("First-time PDF extraction and indexing can take several minutes. The index is reused while it remains available and the documents stay unchanged.")
        bar = st.progress(0, text="Preparing research")
        previous_phase = None
        def update(message, fraction):
            nonlocal previous_phase
            bar.progress(min(max(fraction, 0), 1), text=message)
            # Log phase changes, not every page or chunk, and never profile/API data.
            phase = message.split(" · ")[0]
            if phase != previous_phase:
                print(f"[GapHunter] Research: {message}", flush=True)
                previous_phase = phase
        try:
            context = retrieve_context(query, top_k, categories, progress=update)
            status.update(label=f"Research ready · {len(context)} relevant chunks retrieved", state="complete", expanded=False)
            return context
        except Exception as exc:
            status.update(label=f"Research preparation failed · {type(exc).__name__}", state="error", expanded=True)
            raise


def compact_query(*items) -> str:
    return " ".join(json.dumps(x, ensure_ascii=False)[:2500] for x in items)

def run_pipeline(profile: dict, api_key: str):
    st.session_state.agent_activity = {key: {"state": "queued"} for key, *_ in STAGES}
    render_activity()
    profile_a = safe_stage("Understanding your profile", run_profile_agent, profile, api_key)

    problem_ctx = research_context(compact_query(profile, profile_a), 4, ["research", "pakistan", "market"])
    problems = safe_stage("Hunting for problems", run_problem_agent, profile_a, profile, problem_ctx, api_key)

    market_ctx = research_context(compact_query(profile_a, problems), 4, ["market", "research", "pakistan"])
    market = safe_stage("Researching market trends", run_market_agent, profile_a, problems, market_ctx, api_key)

    comp_ctx = research_context(compact_query(problems, market), 5, ["startups", "market", "pakistan"])
    competitors = safe_stage("Investigating existing solutions", run_competitor_agent, problems, market, comp_ctx, api_key)

    gap_ctx = research_context(compact_query(problems, competitors), 4, ["startups", "market", "research", "pakistan"])
    gaps = safe_stage("Searching for gaps", run_gap_agent, profile_a, problems, market, competitors, gap_ctx, api_key)

    innovation = safe_stage("Exploring unique angles", run_innovation_agent, gaps, profile_a, api_key, None)
    red_team = safe_stage("Red-teaming opportunities", run_red_team_agent, innovation, competitors, market, api_key)
    feasibility = safe_stage("Checking feasibility", run_feasibility_agent, innovation, profile, red_team, api_key)
    evaluation = safe_stage("Evaluating trade-offs", run_evaluator_agent, innovation, gaps, competitors, red_team, feasibility, api_key)
    final = safe_stage("Refining opportunities", run_final_agent, profile, profile_a, innovation, gaps, competitors, market, red_team, feasibility, evaluation, api_key)

    return {
        "profile": profile_a, "problems": problems, "market": market,
        "competitors": competitors, "gaps": gaps, "innovation": innovation,
        "red_team": red_team, "feasibility": feasibility,
        "evaluation": evaluation, "final": final,
    }


st.markdown('<div class="gh-hero"><div class="gh-eyebrow">GapHunter · Opportunity Studio</div>'
            '<h1>Find the gap.<br><span>Build what matters.</span></h1>'
            '<p>Turn your skills, interests and local evidence into startup opportunities. '
            'Explore the market, challenge the assumptions, and shape a practical first step.</p>'
            '<span class="gh-chip">10 specialist agents</span><span class="gh-chip cyan">Local research</span>'
            '<span class="gh-chip">Challenge &amp; refine</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="gh-brand"><span class="gh-logo">✦</span><div><strong>GapHunter</strong>'
                '<small>Opportunity Studio</small></div></div>', unsafe_allow_html=True)
    st.caption("A clearer path from curiosity to a testable opportunity.")
    st.divider()
    preview = st.toggle("Preview sample results", key="preview_mode",
                        help="Explore fictional cards and the agent layout without making API calls.")
    st.caption("Fictional examples · no API calls" if preview else "Your analysis uses your profile and local research.")
    st.divider()
    st.subheader("Research library")
    st.caption("Add curated PDF, TXT or MD files to your knowledge-base folders. The index updates when files change.")
    if st.button("Check research library", use_container_width=True):
        try:
            with st.spinner("Checking your local evidence…"):
                sample = retrieve_context("AI startup market technology Pakistan", 3)
            st.success(f"Ready · {len(sample)} relevant chunk(s) retrieved.")
        except Exception as exc:
            st.error(f"RAG initialization failed: {type(exc).__name__}")
    st.divider()
    st.caption("Evidence supports exploration. Every opportunity still needs validation with real users.")

analysis = demo_analysis() if preview else st.session_state.get("analysis")
if preview:
    st.markdown('<div class="gh-demo">◇ Design preview · Fictional examples. No agents or market research were run.</div>', unsafe_allow_html=True)

with st.expander("Your opportunity profile", expanded=not bool(analysis)):
    with st.form("profile_form"):
        st.caption("Start with what you know, what you enjoy, and what you can realistically build.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Your strengths")
            name = st.text_input("Name (optional)")
            role = st.text_input("Current role", placeholder="University student")
            education = st.text_input("Education / background", placeholder="BS Artificial Intelligence")
            skills = st.text_area("Skills", placeholder="Python, AI/ML, web development, robotics")
            interests = st.text_area("Interests", placeholder="Education, automation, AI engineering")
            experience = st.text_area("Experience", placeholder="Projects, internships, domain experience…")
        with c2:
            st.markdown("#### Your starting point")
            resources = st.text_area("Available resources", placeholder="Laptop, four-person team, no hardware")
            budget = st.text_input("Budget", placeholder="Rs. 100,000")
            market_pref = st.selectbox("Preferred market", ["Pakistan", "Global", "Both", "Not sure"])
            target_users = st.text_input("Target users", placeholder="Students, universities")
            business = st.multiselect("Preferred business type", ["SaaS", "Mobile App", "Hardware", "Marketplace", "Developer Tool", "AI Product", "Service", "Not sure"])
            risk = st.select_slider("Risk tolerance", options=["Low", "Medium", "High"], value="Medium")
            problems_interest = st.text_area("Problems / industries of interest", placeholder="Education, automation, robotics")
        submitted = st.form_submit_button("✦ Discover my opportunities", type="primary", use_container_width=True,
                                         disabled=preview)
        st.caption("Provide at least your skills and interests. Your Groq API key must be configured in Streamlit Secrets.")

activity_slot = st.empty()
if not preview:
    render_activity()

if submitted and not preview:
    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets before running the agent pipeline.")
    elif not skills.strip() or not interests.strip():
        st.error("Please provide at least your skills and interests.")
    else:
        profile = {"name": name, "current_role": role, "education": education, "skills": skills,
                   "interests": interests, "experience": experience, "resources": resources,
                   "budget": budget, "preferred_market": market_pref, "target_users": target_users,
                   "business_preferences": business, "risk_tolerance": risk, "problems_or_industries": problems_interest}
        st.session_state.analysis = run_pipeline(profile, api_key)
        st.session_state.profile_input = profile
        st.session_state.opportunity_feedback = {}
        st.session_state.selected_opportunity = 0
        analysis = st.session_state.analysis

if analysis:
    final_items = analysis.get("final", {}).get("opportunities", [])
    final_items = [item for item in final_items if isinstance(item, dict)] if isinstance(final_items, list) else []
    section("Your opportunity shortlist", "Compare, explore, then validate")
    evidence_count = sum(len(item.get("evidence", [])) for item in final_items if isinstance(item.get("evidence", []), list))
    completed = sum("error" not in analysis.get(key, {"error": "Unavailable"}) for key, *_ in STAGES)
    for col, label, number, note in zip(st.columns(3), ["Opportunities", "Evidence entries", "Stages completed"],
                                       [len(final_items), evidence_count, "Sample" if preview else f"{completed}/10"],
                                       ["Candidates to investigate", "Validate sources and relevance", "Fictional layout preview" if preview else "Warnings remain visible below"]):
        with col:
            st.markdown(f'<div class="gh-stat"><small>{esc(label)}</small><strong>{esc(number)}</strong><span>{esc(note)}</span></div>', unsafe_allow_html=True)
    st.write("")
    if not final_items:
        st.info("No structured final opportunities were returned. Open Agent Outputs below to inspect the pipeline.")
    else:
        eval_items = analysis.get("evaluation", {}).get("evaluations", [])
        evals = {item.get("opportunity_name"): item for item in eval_items if isinstance(item, dict)} if isinstance(eval_items, list) else {}
        columns = st.columns(3 if len(final_items) == 3 else 2 if len(final_items) > 1 else 1)
        for i, opp in enumerate(final_items):
            ev = evals.get(opp.get("name"), {})
            with columns[i % len(columns)]:
                with st.container(border=True):
                    st.markdown(f'<div class="gh-card"><div class="gh-card-number">Opportunity {i+1:02d}<span>✦</span></div>'
                                f'<h3>{esc(opp.get("name", f"Opportunity {i+1}"))}</h3><p>{esc(short(opp.get("problem", "")))}</p>'
                                f'<span class="gh-chip">Evidence · {esc(ev.get("evidence_strength", "Unknown"))}</span>'
                                f'<span class="gh-chip cyan">Feasibility · {esc(ev.get("feasibility", "Unknown"))}</span>'
                                f'<div class="gh-mvp"><small>First build</small><p>{esc(short(opp.get("mvp", "Not available"), 150))}</p></div></div>', unsafe_allow_html=True)
                    if st.button("Explore opportunity →", key=f"explore_{i}", use_container_width=True):
                        st.session_state["preview_selected_opportunity" if preview else "selected_opportunity"] = i
        st.caption("Qualitative labels summarize agent analysis. They do not predict startup success.")

        with st.expander("Compare opportunities side by side"):
            rows = []
            for opp in final_items:
                ev = evals.get(opp.get("name"), {})
                rows.append({"Opportunity": opp.get("name"), "Problem fit": ev.get("problem_clarity", "Unknown"),
                             "Differentiation": ev.get("differentiation", "Unknown"), "Feasibility": ev.get("feasibility", "Unknown"),
                             "Evidence": ev.get("evidence_strength", "Unknown")})
            st.dataframe(rows, use_container_width=True, hide_index=True)

        section("Explore the details", "A practical next step for each candidate")
        selection_key = "preview_selected_opportunity" if preview else "selected_opportunity"
        # Cards update the appropriate selector without mixing demo and real runs.
        if st.session_state.get(selection_key, 0) >= len(final_items):
            st.session_state[selection_key] = 0
        selected = st.selectbox("Choose an opportunity", list(range(len(final_items))),
                                format_func=lambda i: final_items[i].get("name", f"Opportunity {i+1}"), key=selection_key)
        opp = final_items[selected]
        st.subheader(opp.get("name", f"Opportunity {selected+1}"))
        tabs = st.tabs(["Overview", "Evidence & uncertainty", "Validation plan", "Challenge & improve"])
        with tabs[0]:
            left, right = st.columns(2)
            with left:
                show_list("The problem", opp.get("problem"))
                show_list("Target users", opp.get("target_users") or opp.get("who_has_problem"))
                show_list("Proposed solution", opp.get("proposed_solution"))
                show_list("Existing solutions", opp.get("existing_solutions"))
                show_list("Major risks", opp.get("major_risks"))
            with right:
                show_list("Observed gap", opp.get("observed_gap"))
                show_list("Unique angle", opp.get("unique_angle"))
                show_list("Why you", opp.get("why_you"))
                show_list("Your first build", opp.get("mvp"))
                show_list("Business model", opp.get("business_model"))
        with tabs[1]:
            show_list("Evidence", opp.get("evidence"))
            show_list("Inferences", opp.get("inferences"))
            show_list("Hypotheses", opp.get("hypotheses"))
            show_list("Research limitations", analysis.get("final", {}).get("overall_research_limitations"))
        with tabs[2]:
            experiments = opp.get("validation_experiments", [])
            for i, experiment in enumerate(experiments if isinstance(experiments, list) else []):
                if not isinstance(experiment, dict):
                    continue
                with st.container(border=True):
                    st.markdown(f"**Experiment {i+1:02d}**")
                    show_list("Assumption", experiment.get("assumption"))
                    show_list("Test", experiment.get("test"))
                    show_list("Success signal · heuristic", experiment.get("success_signal_heuristic"))
            show_list("What still needs validation", opp.get("what_needs_validation"))
        with tabs[3]:
            st.caption("Pressure-test the assumptions, then use the critique to sharpen the opportunity.")
            api_key = get_api_key()
            disabled = preview or not api_key
            if disabled:
                st.info("Challenge actions become available for your real analysis when a Groq API key is configured.")
            feedback_store = st.session_state.setdefault("opportunity_feedback", {})
            feedback_key = f"{selected}:{opp.get('name', '')}"
            feedback = feedback_store.setdefault(feedback_key, {}) if not preview else {}
            if st.button("◈ Try to kill this idea", key="challenge", use_container_width=True, disabled=disabled):
                feedback["attack"] = safe_stage("Attacking selected opportunity", run_red_team_agent,
                    {"opportunities": [opp]}, analysis.get("competitors", {}), analysis.get("market", {}), api_key)
                feedback.pop("improved", None)
            attack = feedback.get("attack")
            if attack:
                for item in attack.get("attacks", []):
                    if isinstance(item, dict):
                        for field in ("why_it_might_fail", "weak_assumptions", "competitor_threats", "market_risks", "technical_risks", "customer_acquisition_risks", "what_must_be_validated", "severity"):
                            show_list(field.replace("_", " ").title(), item.get(field))
                if "error" in attack or "raw_output" in attack:
                    st.json(attack)
                if st.button("✦ Improve this idea", key="improve", type="primary", use_container_width=True, disabled=disabled):
                    feedback["improved"] = safe_stage("Improving selected opportunity", run_innovation_agent,
                        {"gaps": [{"initial_opportunity": opp}]}, analysis.get("profile", {}), api_key, attack)
            improved = feedback.get("improved")
            if improved:
                st.subheader("Refined opportunity")
                for item in improved.get("opportunities", []):
                    if isinstance(item, dict):
                        for field, value in item.items():
                            show_list(field.replace("_", " ").title(), value)
                if "error" in improved or "raw_output" in improved:
                    st.json(improved)
        st.download_button("↓ Download full analysis (JSON)", data=json.dumps(analysis, ensure_ascii=False, indent=2),
                           file_name="gaphunter_sample.json" if preview else "gaphunter_analysis.json", mime="application/json",
                           use_container_width=True)
    with st.expander("Agent outputs & diagnostics"):
        if preview:
            st.caption("These are fictional sample outputs.")
        st.json(analysis)
else:
    st.markdown('<div class="gh-empty"><span class="gh-chip">Your next step</span>'
                '<h3>Your opportunity shortlist starts here.</h3><p>Complete your profile above, or turn on the sample preview to explore the experience.</p></div>', unsafe_allow_html=True)

if preview:
    section("Meet your research team", "Preview of the ten-stage workflow")
    st.caption("These cards describe each agent's role. No agents are running in this preview.")
    st.markdown(activity_html({}), unsafe_allow_html=True)

st.markdown('<div class="gh-footer">GapHunter AI · Investigate thoughtfully. Validate with real users.</div>', unsafe_allow_html=True)
