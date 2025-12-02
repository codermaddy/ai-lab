# scripts/test_agent.py
import requests

BASE_URL = "http://localhost:8000"

def ask(query: str):
    resp = requests.post(f"{BASE_URL}/agent/query", json={"query": query}, timeout=300)
    print("STATUS:", resp.status_code)
    print("RESPONSE:")
    print(resp.json())

if __name__ == "__main__":
    ask("List my top runs by accuracy.")
    ask("Compare the two best runs and tell me which is better.")
    ask("Flag the best run for publishing.")
    ask("Summarize my experiments on the iris_classification task.")