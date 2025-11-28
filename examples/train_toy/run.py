import time
from pathlib import Path
from lab_logger.core import log_run

@log_run()
def toy_train(lr=0.01, epochs=2):
    for e in range(int(epochs)):
        print(f"[toy] epoch {e+1}/{epochs} lr={lr}")
        time.sleep(0.1)

    # save checkpoint inside the current example directory
    base_dir = Path(__file__).resolve().parent       # examples/train_toy
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"checkpoint_{int(time.time())}.bin"
    with open(fname, "wb") as f:
        f.write(b"fake model")

    return {"status": "ok", "checkpoint": str(fname)}

if __name__ == "__main__":
    toy_train(lr=0.001, epochs=3)
