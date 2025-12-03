# scripts/test_llm_agent.py

import json
import textwrap
import requests

BASE_URL = "http://localhost:8000"


TEST_QUERIES = [
    # 1. Global best runs
    "List my top 10 runs by accuracy.",
    "Compare the two best runs and tell me which is better.",

    # 2. Dataset-scoped
    "List my top 5 runs on iris_classification by accuracy.",
    "Summarize my experiments on wine_classification.",

    # 3. Model-scoped (uses command_substring)
    "Compare my two best logistic_regression runs on iris_classification.",
    "Compare my two best linear_svm runs on breast_cancer_classification.",

    # 4. Tag / note-scoped (if you start tagging with PUBLISH later)
    "Show runs tagged PUBLISH on iris_classification.",

    # 5. Mixed / fuzzy
    "What models seem to work best on breast_cancer_classification in terms of accuracy and runtime?",
]


def pretty_print_response(query: str, resp_json):
    print("=" * 80)
    print(f"QUERY: {query}")
    print("-" * 80)

    if isinstance(resp_json, dict):
        intent = resp_json.get("intent")
        answer = resp_json.get("natural_language_answer")
        used_run_ids = resp_json.get("used_run_ids", [])
        flagged = resp_json.get("flagged_run_id")
        comparison = resp_json.get("comparison")

        print(f"Intent: {intent}")
        print("\nAnswer:")
        print(textwrap.fill(answer or "", width=78))

        print("\nUsed run_ids:")
        if used_run_ids:
            for rid in used_run_ids:
                print(f"  - {rid}")
        else:
            print("  (none)")

        if flagged:
            print(f"\nFlagged run_id: {flagged}")
        else:
            print("\nFlagged run_id: (none)")

        if comparison:
            print("\nComparison object (raw JSON):")
            print(json.dumps(comparison, indent=2))
    else:
        print("Raw response:")
        print(resp_json)

    print()  # blank line


def ask(query: str):
    resp = requests.post(
        f"{BASE_URL}/agent/query",
        json={"query": query},
        timeout=300,
    )
    print(f"HTTP STATUS: {resp.status_code}")

    try:
        resp_json = resp.json()
    except Exception:
        print("Non-JSON response body:")
        print(resp.text)
        return

    pretty_print_response(query, resp_json)


if __name__ == "__main__":
    for q in TEST_QUERIES:
        ask(q)
