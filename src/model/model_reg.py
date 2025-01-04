import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml

# Initialize Dagshub with MLflow
dagshub.init(repo_owner='prathamesh.khade20', repo_name='Watre_test', mlflow=True)

# Set the experiment name
mlflow.set_experiment("Final_model")  # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/prathamesh.khade20/Watre_test.mlflow")

# Load the run ID and model name from the saved JSON file
reports_path = "/workspaces/Watre_test/reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id']  # Fetch run id from the JSON file
model_name = run_info['model_name']  # Fetch model name from the JSON file

# Start the MLflow run
with mlflow.start_run() as run:
    # Log parameters, metrics, and model
    params_path = "params.yaml"
    data_path = "./data/processed/train_processed.csv"

    # Load parameters
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    n_estimators = params["model_building"]["n_estimators"]

    # Load training data
    train_data = pd.read_csv(data_path)
    X_train = train_data.drop(columns=['Potability'], axis=1)
    y_train = train_data['Potability']

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log the model parameters and any metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", model.score(X_train, y_train))  # Example metric

    # Register the model
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, model_name)

    # Get the model version and transition to Staging
    registered_model = client.get_registered_model(model_name)
    latest_version = registered_model.latest_versions[0].version

    # Transition the model to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"Model {model_name} version {latest_version} transitioned to Staging.")
