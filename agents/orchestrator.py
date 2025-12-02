# agents/orchestrator.py

import json
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from tools.langchain_tools import (
    list_runs,
    get_run_details,
    compare_runs,
    flag_run_for_publish,
    leaderboard,
    search_run_summaries,
)

PLANNER_PROMPT = """
You are a planner for an experiment-tracking assistant.

You CANNOT access the database directly. Instead, the system can run these TOOLS
for you (they are HTTP wrappers around a backend):

1) list_runs(
      limit: int = 50,
      offset: int = 0,
      dataset_id: Optional[str] = None,
      command_substring: Optional[str] = None,
      note_substring: Optional[str] = None
   )

2) leaderboard(
      metric_name: str,
      top_k: int = 10,
      dataset_id: Optional[str] = None,
      has_artifacts: Optional[bool] = None
   )

3) get_run_details(run_id: str)

4) compare_runs(run_ids: List[str])

5) flag_run_for_publish(run_id: str)

6) search_run_summaries(
      query: str,
      limit: int = 100
   )

Heuristics:

- "best" / "top" runs by a metric => use leaderboard.
- "list", "show runs" => use list_runs (optionally filter by dataset_id, note, etc.).
- "details" of a specific run => use get_run_details.
- "compare" => use compare_runs.
- "flag", "mark for publish" => use flag_run_for_publish.
- Fuzzy high-level questions like "what worked best" or "summarize experiments"
  => use search_run_summaries.

DB shape:

- Metrics commonly include:
  - "accuracy"
  - "f1_macro"
  - "final_train_loss"
  - "final_val_loss"
  - "runtime_sec"

- If the user says "val_accuracy", first plan with metric_name="val_accuracy".
  The system may retry with metric_name="accuracy" if no runs are found.

Your job:

Given the user query, respond with a SINGLE JSON object with exactly these keys:

- "intent": short string like "list_runs", "best_run", "compare_runs", "flag_run", "summarize"
- "tool_name": one of "list_runs", "leaderboard", "get_run_details",
               "compare_runs", "flag_run_for_publish", "search_run_summaries",
               or null if no tool is needed.
- "tool_args": a JSON object containing the exact arguments for that tool.

Rules:
- If no tool is needed, set "tool_name" to null and "tool_args" to an empty object.
- Do NOT include any other top-level keys.
- Output MUST be valid JSON ONLY (no extra text).
"""

ANSWER_PROMPT = """
You are an experiment-tracking assistant for an AI Lab Notebook.

You will receive:
- user_query: the original user query
- tool_name: which tool (if any) was called
- tool_args: the arguments used for the tool call
- tool_output: the JSON returned by that tool, OR null if no tool or an error occurred

Backend semantics (for reference):

- list_runs returns a list of runs, each like:
  - run_id: string
  - timestamp: ISO8601 string
  - git_hash: string
  - command: string
  - dataset_id: string or null
  - wandb_url: string or null
  - final_metrics: dict with keys like
      "accuracy", "f1_macro", "final_train_loss", "final_val_loss", "runtime_sec"
  - note: string or null

- leaderboard returns a list of top runs by the requested metric_name,
  in the same format as list_runs entries.

- get_run_details returns a single run with additional fields:
  - git_diff, python_version, pip_freeze, device_info, params, artifacts, etc.

- compare_runs returns:
  - run_ids: list of strings
  - metric_keys: list of metric names
  - param_keys: list of hyperparameter names
  - runs: list of detailed runs

- flag_run_for_publish returns the updated run detail with note set to "PUBLISH".

- search_run_summaries returns a list of objects:
  - run_id, dataset_id, note, metrics, snippet

Instructions:

1. Use ONLY tool_output for factual details about runs, metrics, and params.
   Do NOT invent run_ids, metrics, or hyperparameters.

2. If tool_output is null or contains an "error" field, clearly state that
   no data was available or that an error occurred.

3. If tool_output is a list of runs, you may:
   - pick the best one by a metric (e.g. highest accuracy),
   - describe a few top runs,
   - or summarize overall performance.

4. If tool_output is from compare_runs, you may create a "comparison" object
   describing which run is better and why.

FINAL OUTPUT FORMAT:

You MUST respond with a SINGLE JSON object with exactly these top-level keys:

- "intent": string, such as "list_runs", "best_run", "compare_runs", "flag_run", "summarize"
- "natural_language_answer": short human-readable answer
- "used_run_ids": list of run_id strings you actually referenced
- "comparison": either null, or a JSON object describing a comparison
- "flagged_run_id": the run_id that was flagged (string), or null if none

Rules:
- JSON ONLY. No markdown, no comments, no extra keys.
- "used_run_ids" must always be present (it can be an empty list).
- "flagged_run_id" must be null if no run was flagged.
"""


class SimpleAgent:
    """
    Matches your existing chat_agent.py expectations:

    - .invoke(query: str) -> JSON string (matching AgentAnswer schema)
    """

    def __init__(self) -> None:
        self.llm = ChatOllama(
            model="llama3.1",
            temperature=0.1,
        )

        self.planner_parser = JsonOutputParser()
        self.answer_parser = JsonOutputParser()

        self.tool_map = {
            "list_runs": list_runs,
            "leaderboard": leaderboard,
            "get_run_details": get_run_details,
            "compare_runs": compare_runs,
            "flag_run_for_publish": flag_run_for_publish,
            "search_run_summaries": search_run_summaries,
        }

        self.planner_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PLANNER_PROMPT),
                    ("human", "{user_query}"),
                ]
            )
            | self.llm
            | self.planner_parser
        )

        self.answer_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", ANSWER_PROMPT),
                    (
                        "human",
                        "User query:\n{user_query}\n\n"
                        "Tool name: {tool_name}\n"
                        "Tool args: {tool_args}\n"
                        "Tool output: {tool_output}\n"
                    ),
                ]
            )
            | self.llm
            | self.answer_parser
        )

    def invoke(self, query: str) -> str:
        """Return a JSON string (for AgentAnswer.model_validate_json)."""
        user_query = query

        # 1) PLAN
        plan = self.planner_chain.invoke({"user_query": user_query})
        if not isinstance(plan, dict):
            plan = {}

        intent = plan.get("intent", "")
        tool_name = plan.get("tool_name", None)
        tool_args = plan.get("tool_args") or {}

        if not isinstance(tool_args, dict):
            tool_args = {}

        # 2) TOOL CALL
        tool_output: Any = None
        if tool_name and tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            try:
                tool_output = tool.invoke(tool_args)

                # Heuristic: fallback val_accuracy -> accuracy if empty.
                if (
                    tool_name == "leaderboard"
                    and isinstance(tool_output, list)
                    and len(tool_output) == 0
                    and tool_args.get("metric_name") == "val_accuracy"
                ):
                    fallback_args = dict(tool_args)
                    fallback_args["metric_name"] = "accuracy"
                    tool_output = self.tool_map["leaderboard"].invoke(fallback_args)
                    tool_args = fallback_args

            except Exception as e:
                tool_output = {"error": str(e)}

        # 3) ANSWER
        answer = self.answer_chain.invoke(
            {
                "user_query": user_query,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_output": tool_output,
            }
        )

        if not isinstance(answer, dict):
            answer = {}

        answer.setdefault("intent", intent or "unknown")
        answer.setdefault(
            "natural_language_answer",
            "I could not form a detailed answer from the available data.",
        )
        answer.setdefault("used_run_ids", [])
        answer.setdefault("comparison", None)
        answer.setdefault("flagged_run_id", None)

        if not isinstance(answer["used_run_ids"], list):
            answer["used_run_ids"] = []

        # Return JSON string for AgentAnswer.model_validate_json
        return json.dumps(answer)


def build_agent() -> SimpleAgent:
    """Factory used by agents/chat_agent.py."""
    return SimpleAgent()