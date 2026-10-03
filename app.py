from __future__ import annotations

import json
import streamlit as st

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

st.set_page_config(page_title="GapHunter AI", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
border-radius: 18px; margin-bottom: 1.2rem;}
.small-muted {opacity: .72; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🔎 GapHunter AI</h1>'
            '<h3>Discover startup opportunities, not just startup ideas.</h3>'
            '<p>An AI-powered multi-agent system that analyzes your skills, interests, '
            'problems and market opportunities to discover, challenge and refine '
            'potential startup opportunities.</p></div>', unsafe_allow_html=True)

def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

def safe_stage(label, fn, *args):
    try:
        with st.status(label, expanded=False) as status:
            result = fn(*args)
            status.update(label=f"✓ {label}", state="complete")
            return result
    except Exception as exc:
        st.warning(f"⚠️ {label} could not be completed. Remaining analysis will use available information.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: stage failed"}

def compact_query(*items) -> str:
    return " ".join(json.dumps(x, ensure_ascii=False)[:2500] for x in items)

def run_pipeline(profile: dict, api_key: str):
    profile_a = safe_stage("Understanding your profile", run_profile_agent, profile, api_key)

    problem_ctx = retrieve_context(compact_query(profile, profile_a), 4, ["research", "pakistan", "market"])
    problems = safe_stage("Hunting for problems", run_problem_agent, profile_a, profile, problem_ctx, api_key)

    market_ctx = retrieve_context(compact_query(profile_a, problems), 4, ["market", "research", "pakistan"])
    market = safe_stage("Researching market trends", run_market_agent, profile_a, problems, market_ctx, api_key)

    comp_ctx = retrieve_context(compact_query(problems, market), 5, ["startups", "market", "pakistan"])
    competitors = safe_stage("Investigating existing solutions", run_competitor_agent, problems, market, comp_ctx, api_key)

    gap_ctx = retrieve_context(compact_query(problems, competitors), 4, ["startups", "market", "research", "pakistan"])
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

def show_list(title, value):
    st.markdown(f"**{title}**")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                st.write(item)
            else:
                st.markdown(f"• {item}")
    elif value:
        st.write(value)
    else:
        st.caption("Not available")

with st.sidebar:
    st.header("Knowledge base")
    st.caption("Place curated PDF, TXT, or MD files inside knowledge_base/market, research, startups, or pakistan. "
               "The FAISS index updates automatically when files change.")
    if st.button("Check local RAG"):
        try:
            sample = retrieve_context("AI startup market technology Pakistan", 3)
            st.success(f"RAG ready. Retrieved {len(sample)} chunk(s).")
        except Exception as exc:
            st.error(f"RAG initialization failed: {type(exc).__name__}")

with st.form("profile_form"):
    st.subheader("Your Opportunity Profile")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Name (optional)")
        role = st.text_input("Current role", placeholder="University student")
        education = st.text_input("Education / background", placeholder="BS Artificial Intelligence")
        skills = st.text_area("Skills", placeholder="Python, AI/ML, Web Development, Robotics")
        interests = st.text_area("Interests", placeholder="AI Engineering, Robotics, Education")
        experience = st.text_area("Experience", placeholder="Projects, internships, domain experience...")
    with c2:
        resources = st.text_area("Available resources", placeholder="Laptop, 4-person team, no hardware")
        budget = st.text_input("Budget", placeholder="Rs. 100,000")
        market_pref = st.selectbox("Preferred market", ["Pakistan", "Global", "Both", "Not sure"])
        target_users = st.text_input("Target users", placeholder="Students, universities")
        business = st.multiselect("Preferred business type",
            ["SaaS", "Mobile App", "Hardware", "Marketplace", "Developer Tool", "AI Product", "Service", "Not sure"])
        risk = st.select_slider("Risk tolerance", options=["Low", "Medium", "High"], value="Medium")
        problems_interest = st.text_area("Problems / industries of interest", placeholder="Education, automation, robotics")
    submitted = st.form_submit_button("🚀 Discover Opportunities", type="primary", use_container_width=True)

if submitted:
    api_key = get_api_key()
    if not api_key:
        st.error('GROQ_API_KEY is missing. Add it to Streamlit Secrets before running the agent pipeline.')
    elif not skills.strip() or not interests.strip():
        st.error("Please provide at least your skills and interests.")
    else:
        profile = {
            "name": name, "current_role": role, "education": education,
            "skills": skills, "interests": interests, "experience": experience,
            "resources": resources, "budget": budget, "preferred_market": market_pref,
            "target_users": target_users, "business_preferences": business,
            "risk_tolerance": risk, "problems_or_industries": problems_interest,
        }
        st.session_state.analysis = run_pipeline(profile, api_key)
        st.session_state.profile_input = profile

analysis = st.session_state.get("analysis")
if analysis:
    final_items = analysis.get("final", {}).get("opportunities", [])
    st.divider()
    st.header("Opportunity Dashboard")

    if not final_items:
        st.info("No structured final opportunities were returned. Open Agent Outputs below to inspect the pipeline.")
    else:
        cols = st.columns(min(3, len(final_items)))
        for i, opp in enumerate(final_items):
            with cols[i % len(cols)]:
                st.subheader(opp.get("name", f"Opportunity {i+1}"))
                st.write(opp.get("problem", ""))
                st.caption(f"Gap: {opp.get('observed_gap', 'Not available')}")
                st.write(f"**MVP:** {opp.get('mvp', 'Not available')}")

        st.subheader("Comparison")
        evals = {x.get("opportunity_name"): x for x in analysis.get("evaluation", {}).get("evaluations", [])}
        rows = []
        for opp in final_items:
            ev = evals.get(opp.get("name"), {})
            rows.append({
                "Opportunity": opp.get("name"),
                "Problem Fit": ev.get("problem_clarity", "Unknown"),
                "Differentiation": ev.get("differentiation", "Unknown"),
                "Feasibility": ev.get("feasibility", "Unknown"),
                "Evidence": ev.get("evidence_strength", "Unknown"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("These qualitative labels summarize agent analysis. They do not predict startup success.")

        selected = st.selectbox("Explore an opportunity", [o.get("name", f"Opportunity {i+1}") for i, o in enumerate(final_items)])
        opp = next((o for o in final_items if o.get("name") == selected), final_items[0])

        st.subheader(selected)
        tabs = st.tabs(["Opportunity", "Evidence & Uncertainty", "Validation", "Challenge & Improve"])
        with tabs[0]:
            show_list("Problem", opp.get("problem"))
            show_list("Target user", opp.get("target_users") or opp.get("who_has_problem"))
            show_list("Proposed solution", opp.get("proposed_solution"))
            show_list("Existing solutions", opp.get("existing_solutions"))
            show_list("Observed gap", opp.get("observed_gap"))
            show_list("Unique angle", opp.get("unique_angle"))
            show_list("Why you", opp.get("why_you"))
            show_list("MVP", opp.get("mvp"))
            show_list("Business model", opp.get("business_model"))
            show_list("Major risks", opp.get("major_risks"))
        with tabs[1]:
            show_list("Evidence", opp.get("evidence"))
            show_list("Inference", opp.get("inferences"))
            show_list("Hypothesis", opp.get("hypotheses"))
            show_list("Research limitations", analysis.get("final", {}).get("overall_research_limitations"))
        with tabs[2]:
            for exp in opp.get("validation_experiments", []):
                with st.container(border=True):
                    st.write(f"**Assumption:** {exp.get('assumption','')}")
                    st.write(f"**Test:** {exp.get('test','')}")
                    st.write(f"**Suggested success signal / heuristic:** {exp.get('success_signal_heuristic','')}")
            show_list("What still needs validation", opp.get("what_needs_validation"))
        with tabs[3]:
            api_key = get_api_key()
            if st.button("🔴 TRY TO KILL THIS IDEA", use_container_width=True):
                attack_input = {"opportunities": [opp]}
                attack = safe_stage("Attacking selected opportunity", run_red_team_agent,
                                    attack_input, analysis.get("competitors", {}), analysis.get("market", {}), api_key)
                st.session_state.selected_attack = attack
            attack = st.session_state.get("selected_attack")
            if attack:
                st.json(attack)
                if st.button("🟢 IMPROVE THIS IDEA", use_container_width=True):
                    base_gap = {"gaps": [{"initial_opportunity": opp}]}
                    improved = safe_stage("Improving selected opportunity", run_innovation_agent,
                                          base_gap, analysis.get("profile", {}), api_key, attack)
                    st.session_state.improved_idea = improved
            if st.session_state.get("improved_idea"):
                st.markdown("### Refined Opportunity")
                st.json(st.session_state.improved_idea)

    with st.expander("Agent outputs / debugging"):
        st.json(analysis)
