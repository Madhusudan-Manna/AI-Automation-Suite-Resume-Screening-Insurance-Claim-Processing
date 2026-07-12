"""
app.py
------
Single Streamlit app covering both assignment tasks:
  Tab 1 -> Q1: AI Resume Screening Assistant (LangChain + RAG)
  Tab 2 -> Q2: Insurance Claim Processing Agent (LangGraph)

Run with:
    streamlit run app.py
"""

import pickle
import streamlit as st

from resume_screener import build_vectorstore, evaluate_resume, compare_resumes, load_resume_text
from claim_agent import process_claim
from llm_utils import get_llm

st.set_page_config(page_title="Madhusudan Manna - AI Assignments", layout="wide")
st.title("AI Assignments - Madhusudan Manna")

tab1, tab2 = st.tabs(["Q1: Resume Screening Assistant", "Q2: Insurance Claim Agent"])


# ---------------------------------------------------------------------------
# Cache the LLM once per session (loading it is the slow part)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_llm():
    return get_llm()


# ---------------------------------------------------------------------------
# TAB 1 - Resume Screening
# ---------------------------------------------------------------------------

with tab1:
    st.header("AI Resume Screening Assistant")
    st.caption("Upload resumes (PDF), paste a Job Description, and get a structured evaluation for each candidate.")

    jd = st.text_area("Job Description", height=150, placeholder="Paste the job description here...")
    uploaded_resumes = st.file_uploader(
        "Upload one or more resumes (PDF)", type="pdf", accept_multiple_files=True
    )

    use_demo = st.checkbox("No PDFs handy? Use the bundled demo data (Manna.pkl)", value=False)

    if st.button("Evaluate Resumes"):
        if not jd.strip():
            st.warning("Please paste a Job Description first.")
        else:
            with st.spinner("Reading resumes and building the search index..."):
                resume_texts = {}

                if uploaded_resumes:
                    for f in uploaded_resumes:
                        with open(f"/tmp/{f.name}", "wb") as out:
                            out.write(f.getbuffer())
                        resume_texts[f.name] = load_resume_text(f"/tmp/{f.name}")

                if use_demo or not resume_texts:
                    with open("Manna.pkl", "rb") as fh:
                        demo = pickle.load(fh)
                    resume_texts.update(demo["sample_resumes"])

                vectorstore = build_vectorstore(resume_texts)
                llm = load_llm()

            with st.spinner("Evaluating candidates..."):
                results = compare_resumes(list(resume_texts.keys()), vectorstore, jd, llm)

            for r in results:
                with st.expander(f"{r.resume_name} - Match score: {r.match_score}/100 ({r.recommendation})"):
                    st.write("**Summary:**", r.summary)
                    st.write("**Matching skills:**", ", ".join(r.matching_skills) or "-")
                    st.write("**Missing skills:**", ", ".join(r.missing_skills) or "-")
                    st.write("**Strengths:**", ", ".join(r.strengths) or "-")
                    st.write("**Weaknesses:**", ", ".join(r.weaknesses) or "-")


# ---------------------------------------------------------------------------
# TAB 2 - Insurance Claim Agent
# ---------------------------------------------------------------------------

with tab2:
    st.header("Insurance Claim Processing Agent")
    st.caption("Fills in a claim, runs it through the LangGraph workflow, and shows the routing decision.")

    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("Claim ID", value="CLM-1001")
        policy_status = st.selectbox("Policy status", ["active", "expired"])
        claim_amount = st.number_input("Claim amount ($)", min_value=0.0, value=5000.0, step=100.0)
    with col2:
        documents_required = st.multiselect(
            "Documents required",
            ["ID Proof", "Police Report", "Medical Report", "Repair Invoice", "Policy Copy"],
            default=["ID Proof", "Policy Copy"],
        )
        documents_provided = st.multiselect(
            "Documents provided by claimant",
            ["ID Proof", "Police Report", "Medical Report", "Repair Invoice", "Policy Copy"],
            default=["ID Proof", "Policy Copy"],
        )

    if st.button("Process Claim"):
        claim = {
            "claim_id": claim_id,
            "policy_status": policy_status,
            "claim_amount": claim_amount,
            "documents_required": documents_required,
            "documents_provided": documents_provided,
        }
        result = process_claim(claim)

        st.subheader(f"Decision: {result['decision'].upper()}")
        st.write(result["decision_reason"])
        st.write("**Summary:**", result["summary"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Documents OK", str(result["documents_ok"]))
        c2.metric("Eligibility OK", str(result["eligibility_ok"]))
        c3.metric("Fraud score", f"{result['fraud_score']}/100")
