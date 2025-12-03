from pathlib import Path
import time
import joblib
import csv
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

# import decorator (installed editable)
from lab_logger.core import log_run


def build_sgd_model(model_name: str, learning_rate: float, random_state: int) -> SGDClassifier:
    """
    Simple SGD-based model factory.
    We use partial_fit so we can do true epoch-wise training.
    """
    name = model_name.lower()

    if name in ["logreg", "logistic", "logistic_regression"]:
        loss = "log_loss"      # logistic regression
    elif name in ["linear_svm", "svm", "hinge"]:
        loss = "hinge"         # linear SVM
    elif name in ["perceptron"]:
        loss = "perceptron"
    else:
        # default to logistic regression
        loss = "log_loss"

    model = SGDClassifier(
        loss=loss,
        learning_rate="constant",
        eta0=float(learning_rate),
        penalty="l2",
        max_iter=1,      # we control epochs with partial_fit loop
        tol=None,
        random_state=int(random_state),
    )
    return model


def _safe_log_loss(model, X, y):
    try:
        # best case: model has predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            scores = model.decision_function(X)
            scores = np.atleast_2d(scores)
            # If binary and decision_function returns shape (n_samples,)
            if scores.ndim == 1 or scores.shape[1] == 1:
                scores = np.column_stack([-scores, scores])
            # softmax across classes
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return float(log_loss(y, proba))
    except Exception:
        return float("nan")


@log_run()
def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
    n_epochs: int = 30,
    learning_rate: float = 0.01,
    model_name: str = "logistic_regression",
    experiment_name: str = "iris_sgd_baseline",
    task_name: str = "iris_classification",
    dataset_name: str = "iris_classification",
):
    """
    Toy classification experiment with:
    - SGD-based models (logistic_regression, linear_svm, perceptron)
    - True epoch-wise training (n_epochs, partial_fit)
    - Per-epoch train & val loss (for plotting)
    - Final accuracy / f1 / losses / runtime_sec
    - Checkpoint + metrics.csv as artifacts
    - Semantic params: experiment_name, model_name, task_name, dataset_name
      (stored in params_json by the logger)
    """

    wall_start = time.time()

    # 1. Load dataset & split
    # dataset_name controls which sklearn toy dataset we use.
    # We use *_classification names so they match what the agent will query.
    if dataset_name == "iris_classification":
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset_name == "wine_classification":
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset_name in ["breast_cancer_classification", "breast_cancer"]:
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        # Fallback: default to iris if unknown dataset_name is given
        X, y = datasets.load_iris(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    classes = np.unique(y)

    # 2. Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 3. Build model
    model = build_sgd_model(model_name, learning_rate=learning_rate, random_state=random_state)

    # 4. Epoch-wise training loop
    epochs = []
    train_losses = []
    val_losses = []

    for epoch in range(1, int(n_epochs) + 1):
        # First call to partial_fit
        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        # Compute train/val loss each epoch
        tr_loss = _safe_log_loss(model, X_train, y_train)
        va_loss = _safe_log_loss(model, X_val, y_val)

        epochs.append(epoch)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

    # 5. Final metrics (accuracy, f1)
    preds_val = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds_val))
    f1 = float(f1_score(y_val, preds_val, average="macro"))

    final_train_loss = float(train_losses[-1]) if train_losses else float("nan")
    final_val_loss = float(val_losses[-1]) if val_losses else float("nan")

    # 6. Save checkpoint & artifact
    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())

    ckpt_path = ckpt_dir / f"{model_name}_model_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "model_name": model_name,
            "experiment_name": experiment_name,
            "task_name": task_name,
            "dataset_name": dataset_name,
        },
        ckpt_path,
    )

    # metrics CSV: epoch, train_loss, val_loss
    metrics_path = ckpt_dir / f"{model_name}_metrics_{timestamp}.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, va in zip(epochs, train_losses, val_losses):
            writer.writerow([e, tr, va])

    wall_end = time.time()
    runtime_sec = float(wall_end - wall_start)

    # 7. Return values for logger
    final_metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "runtime_sec": runtime_sec,
    }

    artifacts = [
        str(ckpt_path),
        str(metrics_path),
    ]

    return {
        "final_metrics": final_metrics,
        "artifacts": artifacts,
        # semantic fields are in kwargs:
        #   experiment_name, model_name, task_name, dataset_name, n_epochs, learning_rate, ...
    }


if __name__ == "__main__":
    train_model(
        test_size=0.25,
        random_state=0,
        n_epochs=30,
        learning_rate=0.01,
        model_name="logistic_regression",
        experiment_name="iris_sgd_multi_model_v1",
        task_name="iris_classification",
        dataset_name="iris_classification",
    )

## old code

# from pathlib import Path
# import time, joblib
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score

# # import decorator (installed editable)
# from lab_logger.core import log_run

# @log_run()
# def train_model(test_size=0.2, random_state=42, max_iter=200):
#     # load dataset
#     X, y = datasets.load_iris(return_X_y=True)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
#     )

#     # preprocessing
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)

#     # model
#     model = LogisticRegression(max_iter=int(max_iter), solver="lbfgs", multi_class="auto")
#     model.fit(X_train, y_train)

#     # eval
#     preds = model.predict(X_val)
#     acc = float(accuracy_score(y_val, preds))
#     f1 = float(f1_score(y_val, preds, average="macro"))

#     # save checkpoint inside this example folder (portable)
#     base_dir = Path(__file__).resolve().parent
#     ckpt_dir = base_dir / "checkpoints"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_path = ckpt_dir / f"model_{int(time.time())}.joblib"

#     # save model + scaler together
#     joblib.dump({"model": model, "scaler": scaler}, ckpt_path)

#     # return metrics and artifact paths so decorator can pick them up
#     return {
#         "final_metrics": {"accuracy": acc, "f1_macro": f1},
#         "artifacts": [str(ckpt_path)]
#     }

# if __name__ == "__main__":
#     # run with default args; or pass as python run.py --test_size 0.25 etc if you add arg parsing
#     train_model(test_size=0.25, random_state=0, max_iter=300)
# examples/train_model/run.py