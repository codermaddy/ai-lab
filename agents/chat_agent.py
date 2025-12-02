# agents/chat_agent.py

from typing import List, Optional, Any

from fastapi import APIRouter
from pydantic import BaseModel

from agents.agents import AgentAnswer, ComparisonResult
from agents.orchestrator import build_agent


class AgentQuery(BaseModel):
    query: str


class AgentResponse(BaseModel):
    intent: str
    natural_language_answer: str
    used_run_ids: List[str]
    comparison: Optional[ComparisonResult] = None
    flagged_run_id: Optional[str] = None


router = APIRouter()
_agent_executor: Optional[Any] = None  # SimpleAgent instance


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


@router.post("/agent/query", response_model=AgentResponse)
def agent_query(payload: AgentQuery) -> AgentResponse:
    """
    Endpoint used by chatbox.

    - Takes a natural-language query.
    - Runs the SimpleAgent (LLM + tools).
    - Parses the JSON output into AgentAnswer (validation).
    """
    agent = get_agent()

    # SimpleAgent.invoke returns a JSON string
    raw_json = agent.invoke(payload.query)

    # Let Pydantic validate & parse this.
    parsed = AgentAnswer.model_validate_json(raw_json)

    return AgentResponse(
        intent=parsed.intent,
        natural_language_answer=parsed.natural_language_answer,
        used_run_ids=parsed.used_run_ids,
        comparison=parsed.comparison,
        flagged_run_id=parsed.flagged_run_id,
    )