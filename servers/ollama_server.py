import subprocess
import time
import requests

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_BIN = "/opt/homebrew/bin/ollama"  # update based on `which ollama`
MODEL_NAME = "llama3.1"

def is_ollama_running():
    try:
        # Only "/api/tags" works for checking quickly
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        return True
    except Exception:
        return False

def model_exists():
    """Check if model is already downloaded."""
    try:
        out = subprocess.check_output([OLLAMA_BIN, "list"], text=True)
        return MODEL_NAME in out
    except:
        return False


def start_ollama_server():
    print("Starting Ollama server...")

    # Start Ollama daemon in background
    # NOTE: Do NOT capture stdout/stderr into PIPE — it may block.
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("Waiting for Ollama to be ready...")

    # Wait until server responds
    for _ in range(30):  # give up to 30 seconds
        if is_ollama_running():
            print("✔️ Ollama server is ready!")
            return
        time.sleep(1)

    raise RuntimeError("Ollama server did not start within timeout!")


if __name__ == "__main__":
    if is_ollama_running():
        print("Ollama server already running")
    else:
        start_ollama_server()

    print("\nPulling model: llama3.1...")

    if not model_exists():
        print(f"Pulling model {MODEL_NAME} (first-time download)...")
        subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME])
    else:
        print(f"Model {MODEL_NAME} already installed — no download needed.")

    print("Model pulled and ready to use!")