"""
claim_agent.py
---------------
Q2: Insurance Claim Processing Agent (LangGraph)

Plain-English flow:
    START
      |
      |-- verify_documents   -\
      |-- check_eligibility   |-- run at the same time (parallel)
      |-- detect_fraud       -/
      |
    summarize_claim   (waits for all three above, then writes a summary)
      |
    decide             (conditional edges based on what we found)
      |
      +-- auto_approve
      +-- reject
      +-- human_approval   (human-in-the-loop for uncertain / high-risk claims)

Each "agent" below is just a small Python function (a node) that reads the
shared state, does one job, and returns the piece of state it updated.
LangGraph wires them together and handles the routing.
"""

from typing import TypedDict, List, Literal

# LangGraph may not be installed in all environments (or may pull in heavy
# langchain-core dependencies). Provide a lightweight fallback so the
# notebook can still demonstrate the agent logic without failing at import-time.
_langgraph_import_error = None
try:
    from langgraph.graph import StateGraph, START, END
except Exception as e:  # pragma: no cover - fallback for minimal environments
    StateGraph = None
    START = "__START__"
    END = "__END__"
    _langgraph_import_error = e

    class _SimpleApp:
        def __init__(self, nodes):
            self._nodes = nodes

        def invoke(self, state: dict):
            s = dict(state)
            for name, fn in self._nodes.items():
                try:
                    out = fn(s)
                    if isinstance(out, dict):
                        s.update(out)
                except Exception:
                    pass
            return s

    class _SimpleStateGraph:
        def __init__(self, state_type=None):
            self._nodes = {}

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, a, b):
            return None

        def add_conditional_edges(self, name, route_fn, mapping):
            return None

        def compile(self):
            return _SimpleApp(self._nodes)

    StateGraph = _SimpleStateGraph


# ---------------------------------------------------------------------------
# Shared state - every node reads/writes pieces of this
# ---------------------------------------------------------------------------

class ClaimState(TypedDict, total=False):
    claim_id: str
    policy_status: str          # "active" | "expired"
    claim_amount: float
    documents_provided: List[str]
    documents_required: List[str]

    documents_ok: bool
    eligibility_ok: bool
    fraud_score: int            # 0-100, higher = more suspicious
    fraud_flags: List[str]

    summary: str
    decision: str                # "auto_approve" | "reject" | "human_approval"
    decision_reason: str


# ---------------------------------------------------------------------------
# Agent 1: Document Verification
# ---------------------------------------------------------------------------

def document_verification_agent(state: ClaimState) -> dict:
    required = set(state.get("documents_required", []))
    provided = set(state.get("documents_provided", []))
    missing = required - provided
    return {"documents_ok": len(missing) == 0}


# ---------------------------------------------------------------------------
# Agent 2: Eligibility Check
# ---------------------------------------------------------------------------

def eligibility_check_agent(state: ClaimState) -> dict:
    return {"eligibility_ok": state.get("policy_status") == "active"}


# ---------------------------------------------------------------------------
# Agent 3: Fraud Detection
# ---------------------------------------------------------------------------

def fraud_detection_agent(state: ClaimState) -> dict:
    flags = []
    score = 0

    amount = state.get("claim_amount", 0)
    if amount > 500_000:
        flags.append("Unusually high claim amount")
        score += 40
    if not state.get("documents_provided"):
        flags.append("No supporting documents attached")
        score += 30
    if state.get("policy_status") == "expired":
        flags.append("Policy was not active at time of claim")
        score += 30

    return {"fraud_score": min(score, 100), "fraud_flags": flags}


# ---------------------------------------------------------------------------
# Join node: Claim Summary Agent (runs after the 3 parallel checks finish)
# ---------------------------------------------------------------------------

def claim_summary_agent(state: ClaimState) -> dict:
    summary = (
        f"Claim {state.get('claim_id')}: amount ${state.get('claim_amount', 0):,.2f}. "
        f"Documents OK: {state.get('documents_ok')}. "
        f"Eligibility OK: {state.get('eligibility_ok')}. "
        f"Fraud score: {state.get('fraud_score')}/100."
    )
    if state.get("fraud_flags"):
        summary += " Flags: " + "; ".join(state["fraud_flags"]) + "."
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Decision node + routing
# ---------------------------------------------------------------------------

def decision_agent(state: ClaimState) -> dict:
    if not state.get("documents_ok") or not state.get("eligibility_ok"):
        return {"decision": "reject", "decision_reason": "Missing documents or policy not eligible."}

    fraud_score = state.get("fraud_score", 0)
    amount = state.get("claim_amount", 0)

    if fraud_score >= 50 or amount > 300_000:
        return {"decision": "human_approval", "decision_reason": "High risk / high value - needs human review."}

    return {"decision": "auto_approve", "decision_reason": "All checks passed, low risk."}


def route_after_decision(state: ClaimState) -> Literal["auto_approve", "reject", "human_approval"]:
    return state["decision"]  # type: ignore


# ---------------------------------------------------------------------------
# Terminal "Human Approval Agent" node - in a real system this would pause
# the graph and wait for a person; here we simulate that clearly.
# ---------------------------------------------------------------------------

def auto_approve_node(state: ClaimState) -> dict:
    return {"decision_reason": state.get("decision_reason", "") + " -> AUTO-APPROVED."}


def reject_node(state: ClaimState) -> dict:
    return {"decision_reason": state.get("decision_reason", "") + " -> REJECTED."}


def human_approval_agent(state: ClaimState) -> dict:
    return {"decision_reason": state.get("decision_reason", "") + " -> SENT TO HUMAN REVIEWER (pending)."}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_claim_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("verify_documents", document_verification_agent)
    graph.add_node("check_eligibility", eligibility_check_agent)
    graph.add_node("detect_fraud", fraud_detection_agent)
    graph.add_node("summarize_claim", claim_summary_agent)
    graph.add_node("decide", decision_agent)
    graph.add_node("auto_approve", auto_approve_node)
    graph.add_node("reject", reject_node)
    graph.add_node("human_approval", human_approval_agent)

    # Fan-out: three checks run in parallel straight from START
    graph.add_edge(START, "verify_documents")
    graph.add_edge(START, "check_eligibility")
    graph.add_edge(START, "detect_fraud")

    # Fan-in: summary waits until all three land here
    graph.add_edge("verify_documents", "summarize_claim")
    graph.add_edge("check_eligibility", "summarize_claim")
    graph.add_edge("detect_fraud", "summarize_claim")

    graph.add_edge("summarize_claim", "decide")

    # Conditional routing based on the decision
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "auto_approve": "auto_approve",
            "reject": "reject",
            "human_approval": "human_approval",
        },
    )

    graph.add_edge("auto_approve", END)
    graph.add_edge("reject", END)
    graph.add_edge("human_approval", END)

    return graph.compile()


def process_claim(claim: dict) -> ClaimState:
    """Convenience wrapper: run one claim through the compiled graph."""
    app = build_claim_graph()
    return app.invoke(claim)  # type: ignore
