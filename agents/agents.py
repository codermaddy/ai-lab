"""
agent_module.py

This is "Agent & Retrieval" module.

Responsibilities:
- Define AgentAnswer schema (used by frontend).
- Wrap DB FastAPI endpoints as LangChain tools.
- Build a small RAG retriever over run summaries.
- Create a LangChain agent that:
  - Uses tools for factual data.
  - Uses retriever for contextual summaries.
  - Applies safety constraints to avoid hallucination.
- Expose a FastAPI endpoint /agent/query for the chatbox.
"""

from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

# ---- 1. Shared schemas (for agent output) -----------------------------------

class RunSummary(BaseModel):
    run_id: str
    dataset: Optional[str] = None
    model_name: Optional[str] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    run_a: RunSummary
    run_b: RunSummary
    metric_diffs: Dict[str, float] = Field(
        default_factory=dict,
        description="Diffs like {'val_loss': -0.12, 'accuracy': +0.03}",
    )
    hyperparam_diffs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-hparam values for each run, e.g. {'learning_rate': {'a':1e-3,'b':3e-4}}",
    )

########### ------ To be shared to UI ------------- ################
class AgentAnswer(BaseModel):
    intent: str  # "compare_runs", "list_runs", "best_run", "flag_run", etc.
    natural_language_answer: str
    used_run_ids: List[str] = Field(default_factory=list)
    comparison: Optional[ComparisonResult] = None
    flagged_run_id: Optional[str] = None