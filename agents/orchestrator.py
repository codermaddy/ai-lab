from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from tools.langchain_tools import list_runs_tool, get_run_details_tool, compare_runs_tool, flag_run_for_publish_tool

def build_agent() -> AgentExecutor:
    """
    Create the LangChain agent that:
    - Uses tools for factual information.
    - Uses retriever tool for high-level context.
    - Returns replies that will be post-processed into AgentAnswer.
    """

    llm = ChatOllama(
        model="llama3.1",  # or any local model you pulled
        temperature=0.1,
    )

    tools = [
        list_runs_tool,
        get_run_details_tool,
        compare_runs_tool,
        flag_run_for_publish_tool,
        # search_run_summaries_tool,
    ]

    system_prompt = """
You are an experiment-tracking assistant for ML researchers.

You have tools that can:
- list runs,
- fetch run details,
- compare runs,
- search textual run summaries,
- flag runs for publishing.

CRITICAL RULES:
- Do NOT invent run IDs, metrics, hyperparameters, or dates.
- For any factual question about experiments, you MUST call the appropriate tools.
- If tools return no data, clearly state that you couldn't find relevant runs.
- When user asks to 'flag' or 'prepare' a run for publishing, always call the `flag_run_for_publish` tool.
- Use concise, bullet-style summaries by default.
- Your final output MUST be a JSON object with keys:
  - intent (string),
  - natural_language_answer (string),
  - used_run_ids (list of strings),
  - comparison (object or null),
  - flagged_run_id (string or null).

The comparison object, if present, MUST match:
{
  "run_a": {...RunSummary...},
  "run_b": {...RunSummary...},
  "metric_diffs": {...},
  "hyperparam_diffs": {...}
}

Never include any extra top-level keys.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
