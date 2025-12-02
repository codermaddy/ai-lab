class AgentQuery(BaseModel):
    query: str


class AgentResponse(BaseModel):
    """
    This is what frontend receives.

    It's just AgentAnswer validated and re-emitted.
    """
    intent: str
    natural_language_answer: str
    used_run_ids: List[str]
    comparison: Optional[ComparisonResult] = None
    flagged_run_id: Optional[str] = None


router = APIRouter()
_agent_executor: Optional[AgentExecutor] = None


def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


@router.post("/agent/query", response_model=AgentResponse)
def agent_query(payload: AgentQuery) -> AgentResponse:
    """
    Endpoint used by chatbox.

    - Takes a natural-language query.
    - Runs the agent with tools.
    - Parses the JSON output into AgentAnswer (validation).
    """
    agent = get_agent()

    # Run the agent; it will return a string (JSON) in 'output'
    result = agent.invoke({"input": payload.query})
    raw_text = result["output"]

    # Let Pydantic validate & parse this.
    parsed = AgentAnswer.model_validate_json(raw_text)

    return AgentResponse(
        intent=parsed.intent,
        natural_language_answer=parsed.natural_language_answer,
        used_run_ids=parsed.used_run_ids,
        comparison=parsed.comparison,
        flagged_run_id=parsed.flagged_run_id,
    )
