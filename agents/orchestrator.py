# agents/orchestrator.py

import json
from typing import Any, Dict

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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

1) list_runs
   Arguments:
     - limit: int (default 50)
     - offset: int (default 0)
     - dataset_id: optional string
     - command_substring: optional string
     - note_substring: optional string

2) leaderboard
   Arguments:
     - metric_name: string (e.g. "accuracy", "f1_macro")
     - top_k: int (default 10)
     - dataset_id: optional string
     - has_artifacts: optional bool

3) get_run_details
   Arguments:
     - run_id: string

4) compare_runs
   Arguments:
     - run_ids: list of run_id strings

5) flag_run_for_publish
   Arguments:
     - run_id: string

6) search_run_summaries
   Arguments:
     - query: string
     - limit: int (default 100)


REQUIRED TOOL USAGE

If the user is asking about runs, experiments, datasets, models, metrics,
performance, accuracy, loss, F1, or anything that depends on stored run data,
you MUST choose one of the tools:

- list_runs
- leaderboard
- get_run_details
- compare_runs
- flag_run_for_publish
- search_run_summaries

In these cases, tool_name must NOT be null.

Only set tool_name to null when the user asks a purely conceptual question
that does not require any stored run data, such as:
- "Explain what accuracy means in general."
- "What is F1 score?"


MAPPING COMMON QUERIES TO TOOLS

Use these heuristics:

- If the user asks for "best" or "top" runs by a metric:
  Use leaderboard.

  Example language:
  - "List my top 10 runs by accuracy."
    => intent = "best_run"
       tool_name = "leaderboard"
       tool_args should include:
         metric_name = "accuracy"
         top_k = 10

  - "What are my best runs on iris_classification by accuracy?"
    => same as above, but tool_args should also include:
         dataset_id = "iris_classification"

- If the user says "list" or "show runs":
  Use list_runs (optionally filtered by dataset_id or note_substring).

  Example language:
  - "Show runs tagged PUBLISH on iris_classification."
    => intent = "list_runs"
       tool_name = "list_runs"
       tool_args should include:
         limit = 50
         offset = 0
         dataset_id = "iris_classification"
         note_substring = "PUBLISH"

- If the user asks for "details" of a specific run and gives a run ID:
  Use get_run_details.

- If the user says "compare" and mentions "best" or "top" runs:
  If no explicit run IDs are given:
    Use leaderboard with metric_name = "accuracy" (or another metric if clearly asked),
    and a reasonably large top_k such as 10.
    The answering stage can focus on the two best runs from that result.

  If explicit run IDs are given:
    Use compare_runs with run_ids set to those IDs.

- If the user says "flag" or "mark for publish":
  Use flag_run_for_publish with the specified run_id.

- If the user asks fuzzy, high-level questions like:
    "What worked best overall?"
    "Summarize my experiments on breast_cancer_classification."
  Use search_run_summaries with the user query as the search text.


DB SHAPE (FOR YOUR REASONING ONLY)

Typical metrics include:
- accuracy
- f1_macro
- final_train_loss
- final_val_loss
- runtime_sec

If the user says "val_accuracy", prefer metric_name = "val_accuracy".
If no runs are found by the backend, the system may retry with metric_name = "accuracy".


OUTPUT FORMAT

Given the user query, respond with a SINGLE JSON object with exactly these keys:

- "intent": a short string like "list_runs", "best_run", "compare_runs", "flag_run", "summarize"
- "tool_name": one of:
    "list_runs", "leaderboard", "get_run_details",
    "compare_runs", "flag_run_for_publish", "search_run_summaries",
    or null if no tool is needed.
- "tool_args": a JSON object (dictionary) containing the exact arguments for that tool.

Rules:
- If no tool is needed, set "tool_name" to null and "tool_args" to an empty object.
- Do NOT include any other top-level keys.
- Output MUST be valid JSON ONLY, with no extra commentary, no markdown, and no trailing text.
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
  in the same format as list_runs entries. It may already be filtered by
  dataset_id or has_artifacts based on tool_args.

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
   Make sure your description is consistent with tool_args
   (e.g., dataset_id filters, tags, or other constraints).

4. If tool_output is from compare_runs, you may create a "comparison" object
   describing which run is better and why, based on the metrics and parameters.

5. If tool_output is from flag_run_for_publish, explain which run was flagged
   and why it is a reasonable choice (e.g., high accuracy, low loss, good runtime).

6. For search_run_summaries, synthesize a concise narrative:
   - what models or settings tend to work well,
   - what tends to fail,
   - any notable trade-offs (accuracy vs runtime, etc.).

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

# PLANNER_PROMPT = """
# You are a planner for an experiment-tracking assistant.

# You CANNOT access the database directly. Instead, the system can run these TOOLS
# for you (they are HTTP wrappers around a backend):

# 1) list_runs(
#       limit: int = 50,
#       offset: int = 0,
#       dataset_id: Optional[str] = None,
#       command_substring: Optional[str] = None,
#       note_substring: Optional[str] = None
#    )

# 2) leaderboard(
#       metric_name: str,
#       top_k: int = 10,
#       dataset_id: Optional[str] = None,
#       has_artifacts: Optional[bool] = None
#    )

# 3) get_run_details(run_id: str)

# 4) compare_runs(run_ids: List[str])

# 5) flag_run_for_publish(run_id: str)

# 6) search_run_summaries(
#       query: str,
#       limit: int = 100
#    )

# Heuristics:

# - "best" / "top" runs by a metric => use leaderboard.
# - "list", "show runs" => use list_runs (optionally filter by dataset_id, note, etc.).
# - "details" of a specific run => use get_run_details.
# - "compare" => use compare_runs.
# - "flag", "mark for publish" => use flag_run_for_publish.
# - Fuzzy high-level questions like "what worked best" or "summarize experiments"
#   => use search_run_summaries.

# DB shape:

# - Metrics commonly include:
#   - "accuracy"
#   - "f1_macro"
#   - "final_train_loss"
#   - "final_val_loss"
#   - "runtime_sec"

# - If the user says "val_accuracy", first plan with metric_name="val_accuracy".
#   The system may retry with metric_name="accuracy" if no runs are found.

# Your job:

# Given the user query, respond with a SINGLE JSON object with exactly these keys:

# - "intent": short string like "list_runs", "best_run", "compare_runs", "flag_run", "summarize"
# - "tool_name": one of "list_runs", "leaderboard", "get_run_details",
#                "compare_runs", "flag_run_for_publish", "search_run_summaries",
#                or null if no tool is needed.
# - "tool_args": a JSON object containing the exact arguments for that tool.

# Rules:
# - If no tool is needed, set "tool_name" to null and "tool_args" to an empty object.
# - Do NOT include any other top-level keys.
# - Output MUST be valid JSON ONLY (no extra text).
# """

# ANSWER_PROMPT = """
# You are an experiment-tracking assistant for an AI Lab Notebook.

# You will receive:
# - user_query: the original user query
# - tool_name: which tool (if any) was called
# - tool_args: the arguments used for the tool call
# - tool_output: the JSON returned by that tool, OR null if no tool or an error occurred

# Backend semantics (for reference):

# - list_runs returns a list of runs, each like:
#   - run_id: string
#   - timestamp: ISO8601 string
#   - git_hash: string
#   - command: string
#   - dataset_id: string or null
#   - wandb_url: string or null
#   - final_metrics: dict with keys like
#       "accuracy", "f1_macro", "final_train_loss", "final_val_loss", "runtime_sec"
#   - note: string or null

# - leaderboard returns a list of top runs by the requested metric_name,
#   in the same format as list_runs entries.

# - get_run_details returns a single run with additional fields:
#   - git_diff, python_version, pip_freeze, device_info, params, artifacts, etc.

# - compare_runs returns:
#   - run_ids: list of strings
#   - metric_keys: list of metric names
#   - param_keys: list of hyperparameter names
#   - runs: list of detailed runs

# - flag_run_for_publish returns the updated run detail with note set to "PUBLISH".

# - search_run_summaries returns a list of objects:
#   - run_id, dataset_id, note, metrics, snippet

# Instructions:

# 1. Use ONLY tool_output for factual details about runs, metrics, and params.
#    Do NOT invent run_ids, metrics, or hyperparameters.

# 2. If tool_output is null or contains an "error" field, clearly state that
#    no data was available or that an error occurred.

# 3. If tool_output is a list of runs, you may:
#    - pick the best one by a metric (e.g. highest accuracy),
#    - describe a few top runs,
#    - or summarize overall performance.

# 4. If tool_output is from compare_runs, you may create a "comparison" object
#    describing which run is better and why.

# FINAL OUTPUT FORMAT:

# You MUST respond with a SINGLE JSON object with exactly these top-level keys:

# - "intent": string, such as "list_runs", "best_run", "compare_runs", "flag_run", "summarize"
# - "natural_language_answer": short human-readable answer
# - "used_run_ids": list of run_id strings you actually referenced
# - "comparison": either null, or a JSON object describing a comparison
# - "flagged_run_id": the run_id that was flagged (string), or null if none

# Rules:
# - JSON ONLY. No markdown, no comments, no extra keys.
# - "used_run_ids" must always be present (it can be an empty list).
# - "flagged_run_id" must be null if no run was flagged.
# """
class SimpleAgent:
    """
    Matches your existing chat_agent.py expectations:

    - .invoke(query: str) -> JSON string (matching AgentAnswer schema)
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
        model="gpt-4.1-mini",
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

    def _fallback_plan(self, user_query: str, original_intent: str) -> Dict[str, Any]:
        """
        Heuristic router if the planner does not pick a valid tool.

        Returns a dict with keys: intent, tool_name, tool_args
        """
        q_lower = user_query.lower()

        def contains_any(words):
            return any(w in q_lower for w in words)

        # Dataset hints we know about
        dataset_hint = None
        for ds in [
            "iris_classification",
            "wine_classification",
            "breast_cancer_classification",
        ]:
            if ds in q_lower:
                dataset_hint = ds
                break

        tool_name: str | None = None
        tool_args: Dict[str, Any] = {}
        intent = original_intent

        # "Compare the two best runs" etc.
        if contains_any(["compare"]) and contains_any(["best", "top"]):
            tool_name = "leaderboard"
            tool_args["metric_name"] = "accuracy"
            tool_args["top_k"] = 10
            if dataset_hint:
                tool_args["dataset_id"] = dataset_hint
            intent = intent or "compare_runs"

        # "top runs", "best runs", "top 10 by accuracy" etc.
        elif contains_any(["top", "best"]) and contains_any(["accuracy", "runs", "run"]):
            tool_name = "leaderboard"
            tool_args["metric_name"] = "accuracy"
            tool_args["top_k"] = 10
            if dataset_hint:
                tool_args["dataset_id"] = dataset_hint
            intent = intent or "best_run"

        # general "list/show runs"
        elif contains_any(["list", "show"]) and contains_any(["runs", "run"]):
            tool_name = "list_runs"
            tool_args["limit"] = 50
            tool_args["offset"] = 0
            if dataset_hint:
                tool_args["dataset_id"] = dataset_hint
            intent = intent or "list_runs"

        # "summarize my experiments ..."
        elif contains_any(["summarize", "summary"]):
            tool_name = "search_run_summaries"
            tool_args["query"] = user_query
            tool_args["limit"] = 100
            intent = intent or "summarize"

        # else: no fallback, leave tool_name = None

        return {
            "intent": intent,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    def invoke(self, query: str) -> str:
        """Return a JSON string (for AgentAnswer.model_validate_json)."""
        user_query = query

        # 1) PLAN with LLM
        plan = self.planner_chain.invoke({"user_query": user_query})
        if not isinstance(plan, dict):
            plan = {}

        intent = plan.get("intent", "") or ""
        tool_name = plan.get("tool_name")
        tool_args = plan.get("tool_args") or {}

        if not isinstance(tool_args, dict):
            tool_args = {}

        # DEBUG: raw planner output
        print("\n[PLANNER] user_query:", user_query)
        print("[PLANNER] raw plan:", plan)

        # 1b) FALLBACK if planner didn't choose a valid tool
        if not tool_name or tool_name not in self.tool_map:
            fb = self._fallback_plan(user_query, intent)
            intent = fb["intent"]
            tool_name = fb["tool_name"]
            tool_args = fb["tool_args"]

        print("[PLANNER] final intent:", intent)
        print("[PLANNER] final tool_name:", tool_name)
        print("[PLANNER] final tool_args:", tool_args)

        # 2) TOOL CALL
        tool_output: Any = None
        if tool_name and tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            try:
                tool_output = tool.invoke(tool_args)
                print("[TOOL] output (truncated):", str(tool_output)[:300])

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
                print("[TOOL] error:", e)
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


# class SimpleAgent:
#     """
#     Matches your existing chat_agent.py expectations:

#     - .invoke(query: str) -> JSON string (matching AgentAnswer schema)
#     """

#     def __init__(self) -> None:
#         # self.llm = ChatOllama(
#         #     model="llama3.1",
#         #     temperature=0.1,
#         # )
#         self.llm = ChatOpenAI(
#         model="gpt-4.1-mini",
#         temperature=0.1,
#         )



#         self.planner_parser = JsonOutputParser()
#         self.answer_parser = JsonOutputParser()

#         self.tool_map = {
#             "list_runs": list_runs,
#             "leaderboard": leaderboard,
#             "get_run_details": get_run_details,
#             "compare_runs": compare_runs,
#             "flag_run_for_publish": flag_run_for_publish,
#             "search_run_summaries": search_run_summaries,
#         }

#         self.planner_chain = (
#             ChatPromptTemplate.from_messages(
#                 [
#                     ("system", PLANNER_PROMPT),
#                     ("human", "{user_query}"),
#                 ]
#             )
#             | self.llm
#             | self.planner_parser
#         )

#         self.answer_chain = (
#             ChatPromptTemplate.from_messages(
#                 [
#                     ("system", ANSWER_PROMPT),
#                     (
#                         "human",
#                         "User query:\n{user_query}\n\n"
#                         "Tool name: {tool_name}\n"
#                         "Tool args: {tool_args}\n"
#                         "Tool output: {tool_output}\n"
#                     ),
#                 ]
#             )
#             | self.llm
#             | self.answer_parser
#         )

    # def invoke(self, query: str) -> str:
    #     """Return a JSON string (for AgentAnswer.model_validate_json)."""
    #     user_query = query

    #     # 1) PLAN
    #     plan = self.planner_chain.invoke({"user_query": user_query})
    #     if not isinstance(plan, dict):
    #         plan = {}

    #     intent = plan.get("intent", "")
    #     tool_name = plan.get("tool_name", None)
    #     tool_args = plan.get("tool_args") or {}

    #     if not isinstance(tool_args, dict):
    #         tool_args = {}

    #     # 2) TOOL CALL
    #     tool_output: Any = None
    #     if tool_name and tool_name in self.tool_map:
    #         tool = self.tool_map[tool_name]
    #         try:
    #             tool_output = tool.invoke(tool_args)

    #             # Heuristic: fallback val_accuracy -> accuracy if empty.
    #             if (
    #                 tool_name == "leaderboard"
    #                 and isinstance(tool_output, list)
    #                 and len(tool_output) == 0
    #                 and tool_args.get("metric_name") == "val_accuracy"
    #             ):
    #                 fallback_args = dict(tool_args)
    #                 fallback_args["metric_name"] = "accuracy"
    #                 tool_output = self.tool_map["leaderboard"].invoke(fallback_args)
    #                 tool_args = fallback_args

    #         except Exception as e:
    #             tool_output = {"error": str(e)}

    #     # 3) ANSWER
    #     answer = self.answer_chain.invoke(
    #         {
    #             "user_query": user_query,
    #             "tool_name": tool_name,
    #             "tool_args": tool_args,
    #             "tool_output": tool_output,
    #         }
    #     )

    #     if not isinstance(answer, dict):
    #         answer = {}

    #     answer.setdefault("intent", intent or "unknown")
    #     answer.setdefault(
    #         "natural_language_answer",
    #         "I could not form a detailed answer from the available data.",
    #     )
    #     answer.setdefault("used_run_ids", [])
    #     answer.setdefault("comparison", None)
    #     answer.setdefault("flagged_run_id", None)

    #     if not isinstance(answer["used_run_ids"], list):
    #         answer["used_run_ids"] = []

    #     # Return JSON string for AgentAnswer.model_validate_json
    #     return json.dumps(answer)
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

        # ------------------------------------------------------------------
        # FALLBACK: if the planner did not pick a tool, route manually
        # ------------------------------------------------------------------
        q_lower = user_query.lower()

        def contains_any(words):
            return any(w in q_lower for w in words)

        if not tool_name or tool_name not in self.tool_map:
            # Dataset hints we know about
            dataset_hint = None
            for ds in [
                "iris_classification",
                "wine_classification",
                "breast_cancer_classification",
            ]:
                if ds in q_lower:
                    dataset_hint = ds
                    break

            # Start with empty args
            tool_args = {}

            # "Compare the two best runs" etc.
            if contains_any(["compare"]) and contains_any(["best", "top"]):
                tool_name = "leaderboard"
                tool_args["metric_name"] = "accuracy"
                tool_args["top_k"] = 10
                if dataset_hint:
                    tool_args["dataset_id"] = dataset_hint
                intent = intent or "compare_runs"

            # "top runs", "best runs", "top 10 by accuracy" etc.
            elif contains_any(["top", "best"]) and contains_any(["accuracy", "runs", "run"]):
                tool_name = "leaderboard"
                tool_args["metric_name"] = "accuracy"
                tool_args["top_k"] = 10
                if dataset_hint:
                    tool_args["dataset_id"] = dataset_hint
                intent = intent or "best_run"

            # general "list/show runs"
            elif contains_any(["list", "show"]) and contains_any(["runs", "run"]):
                tool_name = "list_runs"
                tool_args["limit"] = 50
                tool_args["offset"] = 0
                if dataset_hint:
                    tool_args["dataset_id"] = dataset_hint
                intent = intent or "list_runs"

            # "summarize my experiments ..."
            elif contains_any(["summarize", "summary"]):
                tool_name = "search_run_summaries"
                tool_args["query"] = user_query
                tool_args["limit"] = 100
                intent = intent or "summarize"

            else:
                # leave tool_name as None → no tool call
                tool_name = None
                tool_args = {}

        # DEBUG (optional, but very helpful)
        print("\n[PLANNER] user_query:", user_query)
        print("[PLANNER] raw plan:", plan)
        print("[PLANNER] final tool_name:", tool_name)
        print("[PLANNER] final tool_args:", tool_args)
        # ------------------------------------------------------------------

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