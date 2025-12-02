from run import train_model

MODELS = ["logistic_regression", "linear_svm", "perceptron"]

for model_name in MODELS:
    for seed in [0, 1, 2]:
        train_model(
            test_size=0.25,
            random_state=seed,
            n_epochs=30,
            learning_rate=0.01,
            model_name=model_name,
            experiment_name=f"iris_{model_name}_sweep_v1",
            task_name=f"Task {seed}",
        )
