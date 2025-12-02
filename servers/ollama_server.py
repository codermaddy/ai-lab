import subprocess
import time
import requests

OLLAMA_HOST = "http://localhost:11434"

def is_ollama_running():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        return True
    except:
        return False

def start_ollama_server():
    print("Starting Ollama server...")
    # Start Ollama daemon in background
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Waiting for Ollama to be ready...")

    # Wait until server responds
    for _ in range(20):
        if is_ollama_running():
            print("Ollama server is ready ")
            return
        time.sleep(1)

    raise RuntimeError("Ollama server did not start within timeout!")

if __name__ == "__main__":
    if is_ollama_running():
        print("Ollama server already running️")
    else:
        start_ollama_server()

    print("\nYou can now run your agent code here!")
    # Example:
    # from agent_core_ollama import run_query
    # print(run_query("Hello from the auto-started Ollama server!"))