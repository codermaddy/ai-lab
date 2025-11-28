from pathlib import Path
import time, joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# import decorator (installed editable)
from lab_logger.core import log_run

@log_run()
def train_model(test_size=0.2, random_state=42, max_iter=200):
    # load dataset
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    # preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # model
    model = LogisticRegression(max_iter=int(max_iter), solver="lbfgs", multi_class="auto")
    model.fit(X_train, y_train)

    # eval
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    f1 = float(f1_score(y_val, preds, average="macro"))

    # save checkpoint inside this example folder (portable)
    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{int(time.time())}.joblib"

    # save model + scaler together
    joblib.dump({"model": model, "scaler": scaler}, ckpt_path)

    # return metrics and artifact paths so decorator can pick them up
    return {
        "final_metrics": {"accuracy": acc, "f1_macro": f1},
        "artifacts": [str(ckpt_path)]
    }

if __name__ == "__main__":
    # run with default args; or pass as python run.py --test_size 0.25 etc if you add arg parsing
    train_model(test_size=0.25, random_state=0, max_iter=300)
