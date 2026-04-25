"""
NEXUS 2.0 — Master LangGraph Workflow
Compiles the SAR generation pipeline: mask → route → typology → narrative → judge → unmask
"""

from langgraph.graph import StateGraph, END

from graph_state import SARState
from nodes import (
    mask_pii_node,
    router_node,
    typology_node,
    narrative_node,
    compliance_judge_node,
    unmask_node,
)


def build_sar_workflow() -> StateGraph:
    """
    Construct and compile the LangGraph workflow for SAR generation.

    Flow:
        mask_pii  →  router  →  typology  →  narrative  →  compliance_judge  →  unmask  →  END

    Returns:
        A compiled LangGraph that can be invoked with SARState.
    """
    graph = StateGraph(SARState)

    # ── Register nodes ──
    graph.add_node("mask_pii", mask_pii_node)
    graph.add_node("router", router_node)
    graph.add_node("typology", typology_node)
    graph.add_node("narrative", narrative_node)
    graph.add_node("compliance_judge", compliance_judge_node)
    graph.add_node("unmask", unmask_node)

    # ── Define edges (linear pipeline) ──
    graph.set_entry_point("mask_pii")
    graph.add_edge("mask_pii", "router")
    graph.add_edge("router", "typology")
    graph.add_edge("typology", "narrative")
    graph.add_edge("narrative", "compliance_judge")
    graph.add_edge("compliance_judge", "unmask")
    graph.add_edge("unmask", END)

    return graph.compile()


# ── Module-level singleton (compiled once, reused per request) ──
sar_pipeline = build_sar_workflow()


def run_pipeline(case_id: str, raw_alert_data: dict) -> SARState:
    """
    Execute the full SAR generation pipeline.

    Args:
        case_id: Unique case identifier.
        raw_alert_data: Raw alert dictionary from the frontend.

    Returns:
        Final SARState with all fields populated.
    """
    initial_state: SARState = {
        "case_id": case_id,
        "raw_alert_data": raw_alert_data,
        "masked_data": {},
        "pii_mapping": {},
        "detected_typology": "",
        "typology_analysis": "",
        "draft_sar_masked": "",
        "final_sar_clean": "",
        "compliance_score": 0,
        "audit_log": [],
        "equilibrium_audit_log": [],
    }

    result = sar_pipeline.invoke(initial_state)
    return result
