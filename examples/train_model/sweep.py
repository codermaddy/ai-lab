from itertools import product

# Support both "python sweep.py" and "python -m examples.train_model.sweep"
try:
    from .run import train_model
except ImportError:
    from run import train_model  # type: ignore


# Datasets to try (these must match what train_model expects in dataset_name)
DATASETS = [
    "iris_classification",
    "wine_classification",
    "breast_cancer_classification",
]

# Models to try – these will appear in the logged command / params
MODELS = ["logistic_regression", "linear_svm", "perceptron"]

# Random seeds
SEEDS = [0, 1, 2]

# Hyperparams
LEARNING_RATES = [0.01, 0.1]
N_EPOCHS = 30
TEST_SIZE = 0.25


def main():
    for dataset_name, model_name, seed, lr in product(DATASETS, MODELS, SEEDS, LEARNING_RATES):
        # experiment_name encodes dataset + model
        experiment_name = f"{dataset_name}_{model_name}_sweep_v1"
        # task_name doubled as dataset identifier (useful for the agent)
        task_name = dataset_name

        print(
            f"[SWEEP] dataset={dataset_name}, model={model_name}, "
            f"seed={seed}, lr={lr}, experiment_name={experiment_name}"
        )

        train_model(
            test_size=TEST_SIZE,
            random_state=seed,
            n_epochs=N_EPOCHS,
            learning_rate=lr,
            model_name=model_name,
            experiment_name=experiment_name,
            task_name=task_name,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    main()
