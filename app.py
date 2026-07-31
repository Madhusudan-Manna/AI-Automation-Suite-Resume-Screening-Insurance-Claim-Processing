"""Streamlit assignment app with both tasks: resume screening (RAG) and insurance claim agent."""

from __future__ import annotations

import os
import pickle
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from claim_agent import process_claim
from llm_utils import get_llm
from resume_screener import build_vectorstore, compare_resumes, load_resume_text

st.set_page_config(page_title="AI Assignments - Madhusudan Manna", layout="wide")
st.title("AI Assignments - Madhusudan Manna")

tab1, tab2 = st.tabs(["Q1: Resume Screening Assistant", "Q2: Insurance Claim Agent"])


@st.cache_resource
def load_llm():
    return get_llm()


with tab1:
    st.header("AI Resume Screening Assistant")
    st.caption("Upload resumes (PDF), paste a job description, and get a structured evaluation for each candidate.")

    jd = st.text_area("Job Description", height=150, placeholder="Paste the job description here...")
    uploaded_resumes = st.file_uploader(
        "Upload one or more resumes (PDF)",
        type="pdf",
        accept_multiple_files=True,
    )
    use_demo = st.checkbox("Use bundled demo resume data", value=False)

    if st.button("Evaluate Resumes"):
        if not jd.strip():
            st.warning("Please paste a Job Description first.")
        else:
            resume_texts = {}
            if uploaded_resumes:
                temp_dir = tempfile.mkdtemp(prefix="resume_")
                try:
                    for uploaded in uploaded_resumes:
                        temp_file_path = os.path.join(temp_dir, uploaded.name)
                        with open(temp_file_path, "wb") as file_obj:
                            file_obj.write(uploaded.getbuffer())
                        resume_texts[uploaded.name] = load_resume_text(temp_file_path)
                finally:
                    for file_name in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file_name))
                    os.rmdir(temp_dir)

            if use_demo or not resume_texts:
                demo_path = "Manna.pkl"
                if os.path.exists(demo_path):
                    with open(demo_path, "rb") as fh:
                        demo = pickle.load(fh)
                    resume_texts.update(demo.get("sample_resumes", {}))
                else:
                    st.warning("No demo file found. Please upload a resume or keep a valid Manna.pkl in the app directory.")
                    st.stop()

            if not resume_texts:
                st.warning("No resume text was extracted. Please upload valid PDF files.")
                st.stop()

            with st.spinner("Building the RAG index and evaluating resumes..."):
                vectorstore = build_vectorstore(resume_texts)
                llm = load_llm()
                results = compare_resumes(list(resume_texts.keys()), vectorstore, jd, llm)

            for result in results:
                with st.expander(f"{result.resume_name} - Match score: {result.match_score}/100 ({result.recommendation})"):
                    st.write("**Summary:**", result.summary)
                    st.write("**Matching skills:**", ", ".join(result.matching_skills) or "-")
                    st.write("**Missing skills:**", ", ".join(result.missing_skills) or "-")
                    st.write("**Strengths:**", ", ".join(result.strengths) or "-")
                    st.write("**Weaknesses:**", ", ".join(result.weaknesses) or "-")


with tab2:
    st.header("Insurance Claim Processing Agent")
    st.caption("Fill in a claim and route it through a LangGraph-based decision workflow.")

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
