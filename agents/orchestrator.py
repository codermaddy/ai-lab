# agents/orchestrator.py

import json
from typing import Any, Callable, Dict

from langchain_ollama import ChatOllama

from tools.langchain_tools import (
    list_runs,
    get_run_details,
    compare_runs,
    flag_run_for_publish,
    leaderboard,
    search_run_summaries,
)
from agents.agents import AgentAnswer


PLANNER_PROMPT = """
You are a planner for an experiment-tracking assistant.

You CANNOT access the database directly. Instead, you have these TOOLS that
the system can run for you:

1) list_runs(
      limit: int = 50,
      offset: int = 0,
      dataset_id: Optional[str] = None,
      command_substring: Optional[str] = None,
      note_substring: Optional[str] = None
   )
   Use when the user asks to list or search runs.

2) leaderboard(
      metric_name: str,
      top_k: int = 10,
      dataset_id: Optional[str] = None,
      has_artifacts: Optional[bool] = None
   )
   Use when the user asks for "best" runs by some metric.

3) get_run_details(run_id: str)
   Use when the user needs exact hyperparameters or metrics for one run.

4) compare_runs(run_ids: List[str])
   Use when the user wants a comparison of specific runs.

5) flag_run_for_publish(run_id: str)
   Use when the user asks to "flag", "mark for publish", etc.

6) search_run_summaries(
      query: str,
      limit: int = 100
   )
   Use for fuzzy high-level questions like
   "What worked best on CIFAR-10?" or "Summarize my experiments this month".

Your job:
Given the user query, decide:
- intent: a short string like "list_runs", "best_run", "compare_runs",
          "flag_run", "summarize", etc.
- tool_name: one of ["list_runs", "leaderboard", "get_run_details",
                     "compare_runs", "flag_run_for_publish",
                     "search_run_summaries"] or null if no tool is needed.
- tool_args: a JSON object with EXACT parameter names for the chosen tool.

You MUST respond with a SINGLE JSON object like:

{
  "intent": "best_run",
  "tool_name": "leaderboard",
  "tool_args": {
    "metric_name": "val_accuracy",
    "top_k": 5,
    "dataset_id": "CIFAR-10",
    "has_artifacts": true
  }
}

If no tool is needed, use: "tool_name": null and "tool_args": {}.
Do NOT include any other keys, and DO NOT add comments.
"""

ANSWER_PROMPT = """
You are an experiment-tracking assistant.

You will receive:
- the user's query
- which tool (if any) was called
- that tool's arguments
- that tool's JSON output (or an error)

You MUST:
- Use ONLY the tool_output data for factual details about runs, metrics, etc.
- NEVER invent run IDs or metric values.
- If tool_output is null or an error occurred, say clearly that no data was available.

At the end, you MUST respond with a SINGLE JSON object that conforms to this schema:

{schema}

Where:
- intent: string (e.g., "list_runs", "best_run", "compare_runs", "flag_run", "summarize")
- natural_language_answer: a short human-readable answer
- used_run_ids: list of run_id strings actually referenced in your answer
- comparison: null OR an object describing a comparison (you may leave it null unless the user asked to compare)
- flagged_run_id: string or null, which run was flagged for publish (if any)

JSON ONLY. No markdown, no comments, no extra keys.
"""


class SimpleAgent:
    """
    Minimal 'agent' that:
    1) Asks LLM to choose a tool + args.
    2) Calls that Python tool.
    3) Asks LLM again to produce final AgentAnswer JSON, grounded on tool output.
    """

    def __init__(self, llm: ChatOllama, tools: Dict[str, Callable[..., Any]]):
        self.llm = llm
        self.tools = tools

    def _call_llm(self, prompt: str) -> str:
        """Call ChatOllama and always return plain text."""
        res = self.llm.invoke(prompt)
        # ChatOllama usually returns an AIMessage-like object with .content
        content = getattr(res, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(res, str):
            return res
        if isinstance(res, dict) and "content" in res:
            return str(res["content"])
        return str(res)

    def _plan(self, user_query: str) -> Dict[str, Any]:
        planning_prompt = f"{PLANNER_PROMPT}\n\nUser query:\n{user_query}\n"
        text = self._call_llm(planning_prompt)
        try:
            plan = json.loads(text)
        except Exception:
            # fallback: no tool, unknown intent
            plan = {"intent": "unknown", "tool_name": None, "tool_args": {}}
        # Normalize keys
        if "tool_args" not in plan or plan["tool_args"] is None:
            plan["tool_args"] = {}
        return plan

    def invoke(self, user_query: str) -> str:
        """
        Main entrypoint.
        Returns: JSON string that should match AgentAnswer schema.
        """
        # 1) Decide which tool to use (if any)
        plan = self._plan(user_query)
        intent = plan.get("intent") or "unknown"
        tool_name = plan.get("tool_name")
        tool_args = plan.get("tool_args") or {}

        tool_output: Any = None
        tool_error: str | None = None

        # 2) Call the tool in Python
        if tool_name and tool_name in self.tools:
            try:
                tool_output = self.tools[tool_name](**tool_args)
            except Exception as e:
                tool_error = f"{type(e).__name__}: {e}"

        # 3) Ask LLM to produce final AgentAnswer JSON
        context = {
            "intent": intent,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_output": tool_output,
            "tool_error": tool_error,
        }

        answer_prompt = ANSWER_PROMPT.format(
            schema=json.dumps(AgentAnswer.model_json_schema(), indent=2)
        )
        final_prompt = (
            answer_prompt
            + "\n\nUser query:\n"
            + user_query
            + "\n\nTool context (JSON):\n"
            + json.dumps(context, indent=2)
            + "\n\nRemember: respond ONLY with the JSON object described above."
        )

        answer_text = self._call_llm(final_prompt)

        # Validate and normalize using Pydantic
        try:
            parsed = AgentAnswer.model_validate_json(answer_text)
            return parsed.model_dump_json()
        except Exception:
            # Fallback minimal answer if LLM returns bad JSON
            fallback = AgentAnswer(
                intent=intent or "unknown",
                natural_language_answer=(
                    "I attempted to plan and call tools, but could not produce a fully "
                    "structured answer. Please try a simpler query."
                ),
                used_run_ids=[],
                comparison=None,
                flagged_run_id=None,
            )
            return fallback.model_dump_json()


def build_agent() -> SimpleAgent:
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.1,
    )

    tools: Dict[str, Callable[..., Any]] = {
        "list_runs": list_runs,
        "leaderboard": leaderboard,
        "get_run_details": get_run_details,
        "compare_runs": compare_runs,
        "flag_run_for_publish": flag_run_for_publish,
        "search_run_summaries": search_run_summaries,
    }

    return SimpleAgent(llm=llm, tools=tools)