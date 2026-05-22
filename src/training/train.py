from pathlib import Path
import yaml
import mlflow


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

def train():
    config = load_config()

    experiment_name = config["experiment"]["name"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    architecture = config["model"]["architecture"]
    dropout_rate = config["model"]["dropout_rate"]

    use_mlflow = config["tracking"]["use_mlflow"]

    if use_mlflow:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("architecture", architecture)
            mlflow.log_param("dropout_rate", dropout_rate)

            # Placeholder metric
            accuracy = 0.93
            mlflow.log_metric("accuracy", accuracy)

            print("Training completed.")
            print(f"Accuracy: {accuracy}")

    else:
        print("MLflow tracking disabled.")


if __name__ == "__main__":
    train()