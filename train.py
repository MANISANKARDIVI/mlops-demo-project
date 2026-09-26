from __future__ import annotations

import argparse
import math
import os
from itertools import product

import mlflow
import mlflow.data
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import LoggedModelStatus
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLflow demo - Wine Quality Prediction"
    )

    parser.add_argument(
        "--csv",
        default="data/wine_sample.csv",
        help="Path to CSV (default: data/wine_sample.csv)",
    )

    parser.add_argument(
        "--target",
        default="quality",
        help="Target column name (default: quality)",
    )

    parser.add_argument(
        "--experiment",
        default="wine-prediction",
        help="MLflow experiment name (default: wine-prediction)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def setup_mlflow_experiment(experiment_name: str):
    """Create or restore an MLflow experiment."""

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)

        print(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")

    elif experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

        print(
            f"Restored deleted MLflow experiment: "
            f"{experiment_name} "
            f"(ID: {experiment.experiment_id})"
        )

    else:
        print(
            f"Using existing MLflow experiment: "
            f"{experiment_name} "
            f"(ID: {experiment.experiment_id})"
        )

    mlflow.set_experiment(experiment_name)


def get_previous_runs(
    experiment_name: str,
    max_results: int = 5,
):
    """Return the most recent previous runs."""

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    return runs


def get_next_run_name(experiment_name: str) -> str:
    """Generate run-1, run-2, run-3, ..."""

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return "run-1"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=50000,
        order_by=["start_time DESC"],
    )

    highest_run_number = 0

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")

        if not run_name.startswith("run-"):
            continue

        try:
            number = int(run_name.split("-", 1)[1])
            highest_run_number = max(highest_run_number, number)
        except ValueError:
            continue

    return f"run-{highest_run_number + 1}"


def print_previous_runs(experiment_name: str):
    """Print recent runs for visibility."""

    runs = get_previous_runs(
        experiment_name,
        max_results=5,
    )

    if not runs:
        print("\nNo previous runs found.")

        return

    print("\n========== PREVIOUS RUNS ==========")

    for run in runs:
        run_name = run.data.tags.get(
            "mlflow.runName",
            run.info.run_id,
        )

        rmse = run.data.metrics.get("best_test_rmse")

        if rmse is None:
            rmse = run.data.metrics.get("test_rmse")

        r2 = run.data.metrics.get("best_test_r2")

        if r2 is None:
            r2 = run.data.metrics.get("test_r2")

        print(
            f"{run_name:<15} "
            f"RMSE={rmse if rmse is not None else 'N/A'} "
            f"R2={r2 if r2 is not None else 'N/A'} "
            f"Run ID={run.info.run_id}"
        )


def search_logged_models(experiment_name: str):
    """Search previously logged MLflow 3 models."""

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    try:
        models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            max_results=20,
            order_by=[
                {
                    "field_name": "creation_time",
                    "ascending": False,
                }
            ],
        )

        return models

    except Exception as exc:
        print(f"Logged model search unavailable: {exc}")

        return []


def print_logged_models(experiment_name: str):
    """Print previously logged models."""

    models = search_logged_models(experiment_name)

    if not models:
        print("\nNo logged models found.")

        return

    print("\n========== LOGGED MODELS ==========")

    for model in models:
        print(f"Name={model.name} ID={model.model_id} Status={model.status}")


def manage_best_logged_model(
    best_model,
    best_trial,
    X_train,
    experiment_name: str,
):
    """
    Create, log, and finalize an MLflow 3 LoggedModel.

    This is only called when:
        MLFLOW_ENABLE_LOGGED_MODEL=true

    Requires a writable artifact store such as S3.
    """

    print("\n========== LOGGED MODEL ==========")

    logged_model = mlflow.initialize_logged_model(
        name="wine-random-forest",
        source_run_id=best_trial["run_id"],
        model_type="sklearn.RandomForestRegressor",
        tags={
            "model_role": "best_trial",
            "dataset": "wine-quality",
            "framework": "scikit-learn",
            "task": "regression",
            "environment": "local",
        },
        params={
            "n_estimators": str(best_trial["n_estimators"]),
            "max_depth": str(best_trial["max_depth"]),
            "random_state": str(best_trial["random_state"]),
        },
    )

    print(f"Initialized logged model: {logged_model.model_id}")

    try:
        signature = infer_signature(
            X_train,
            best_model.predict(X_train),
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            model_id=logged_model.model_id,
            signature=signature,
            input_example=X_train.head(1),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            tags={
                "dataset": "wine-quality",
                "model_role": "best_trial",
            },
        )

        final_model = mlflow.finalize_logged_model(
            model_id=logged_model.model_id,
            status=LoggedModelStatus.READY,
        )

        print(f"Logged model ready: {final_model.model_id}")

        return final_model

    except Exception:
        mlflow.finalize_logged_model(
            model_id=logged_model.model_id,
            status=LoggedModelStatus.FAILED,
        )

        raise


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # MLflow configuration
    # ---------------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    setup_mlflow_experiment(args.experiment)

    print_previous_runs(args.experiment)

    run_name = get_next_run_name(args.experiment)

    enable_logged_model = (
        os.getenv(
            "MLFLOW_ENABLE_LOGGED_MODEL",
            "false",
        ).lower()
        == "true"
    )

    print(f"\nMLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {args.experiment}")
    print(f"MLflow Run Name     : {run_name}")
    print(f"CSV                 : {args.csv}")
    print(f"Logged Model        : {'enabled' if enable_logged_model else 'disabled'}")

    # ---------------------------------------------------------
    # sklearn autologging
    # ---------------------------------------------------------
    #
    # Dataset lineage is manually logged below, so disable
    # automatic dataset logging to avoid duplicate inputs.
    #
    # Model artifact logging remains disabled unless the
    # explicit logged-model feature is enabled.
    #

    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False,
        log_datasets=False,
        exclusive=False,
        silent=False,
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}. "
            "Create or copy wine_sample.csv into the data directory."
        )

    df = pd.read_csv(args.csv)

    print(f"\nDataset shape: {df.shape}")

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------
    # Prepare features and target
    # ---------------------------------------------------------

    columns_to_drop = [args.target]

    if "Id" in df.columns:
        columns_to_drop.append("Id")

    X = df.drop(columns=columns_to_drop)

    y = df[args.target]

    print(f"Features: {list(X.columns)}")

    print(f"Target: {args.target}")

    print(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Training rows: {len(X_train)}")

    print(f"Testing rows : {len(X_test)}")

    # ---------------------------------------------------------
    # Create dataset lineage objects
    # ---------------------------------------------------------

    train_df = X_train.copy()
    train_df[args.target] = y_train

    train_dataset = mlflow.data.from_pandas(
        train_df,
        source=args.csv,
        name="wine-quality-train",
        targets=args.target,
    )

    # ---------------------------------------------------------
    # Parent MLflow run
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=True,
    ) as parent_run:
        # -----------------------------------------------------
        # Tags
        # -----------------------------------------------------

        mlflow.set_tags(
            {
                "run_type": "hyperparameter_tuning",
                "model_type": "RandomForestRegressor",
                "task": "regression",
                "dataset": "wine-quality",
                "framework": "scikit-learn",
                "environment": "local",
                "training_split": "80%",
                "testing_split": "20%",
            }
        )

        # -----------------------------------------------------
        # Note
        # -----------------------------------------------------

        mlflow.set_tag(
            "mlflow.note.content",
            (
                "Wine quality RandomForest regression "
                "experiment using an 80/20 train-test split. "
                "ID column excluded from model features."
            ),
        )

        # -----------------------------------------------------
        # Parameters
        # -----------------------------------------------------

        mlflow.log_params(
            {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X.shape[1],
                "target": args.target,
            }
        )

        # -----------------------------------------------------
        # Dataset lineage
        # -----------------------------------------------------

        mlflow.log_input(
            train_dataset,
            context="training",
            tags={
                "split": "train",
                "source_file": args.csv,
            },
        )

        # -----------------------------------------------------
        # Artifact location
        # -----------------------------------------------------

        artifact_uri = mlflow.get_artifact_uri()

        print(f"MLflow Artifact URI : {artifact_uri}")

        mlflow.set_tag(
            "artifact_uri",
            artifact_uri,
        )

        # -----------------------------------------------------
        # Hyperparameter search
        # -----------------------------------------------------

        param_grid = list(
            product(
                [50, 100],
                [5, 10],
            )
        )

        mlflow.log_param(
            "total_trials",
            len(param_grid),
        )

        print("\n========== HYPERPARAMETER SEARCH ==========")

        print(f"Total trials: {len(param_grid)}")

        best_rmse = float("inf")
        best_trial = None
        best_model = None

        # -----------------------------------------------------
        # Parent trace
        # -----------------------------------------------------

        with mlflow.start_span("train_pipeline") as pipeline_span:
            pipeline_span.set_inputs(
                {
                    "dataset": args.csv,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "feature_count": X.shape[1],
                    "total_trials": len(param_grid),
                }
            )

            # -------------------------------------------------
            # Child runs
            # -------------------------------------------------

            for trial_number, (
                n_estimators,
                max_depth,
            ) in enumerate(
                param_grid,
                start=1,
            ):
                trial_name = f"trial-{trial_number}"

                print(
                    f"\nTraining {trial_name}: "
                    f"n_estimators={n_estimators}, "
                    f"max_depth={max_depth}"
                )

                with mlflow.start_run(
                    run_name=trial_name,
                    nested=True,
                ) as child_run:
                    mlflow.set_tags(
                        {
                            "run_type": "hyperparameter_trial",
                            "trial_number": str(trial_number),
                        }
                    )

                    with mlflow.start_span("model_training") as training_span:
                        training_span.set_inputs(
                            {
                                "model_type": "RandomForestRegressor",
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "train_rows": len(X_train),
                                "feature_count": X.shape[1],
                            }
                        )

                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=args.random_state,
                        )

                        model.fit(
                            X_train,
                            y_train,
                        )

                        training_span.set_outputs(
                            {
                                "status": "completed",
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                            }
                        )

                    # -----------------------------------------
                    # Prediction
                    # -----------------------------------------

                    with mlflow.start_span("prediction") as prediction_span:
                        prediction_span.set_inputs(
                            {
                                "test_rows": len(X_test),
                                "feature_count": X_test.shape[1],
                            }
                        )

                        preds = model.predict(X_test)

                        prediction_span.set_outputs(
                            {
                                "prediction_count": len(preds),
                            }
                        )

                    # -----------------------------------------
                    # Evaluation
                    # -----------------------------------------

                    with mlflow.start_span("evaluation") as evaluation_span:
                        mse = float(
                            mean_squared_error(
                                y_test,
                                preds,
                            )
                        )

                        rmse = float(math.sqrt(mse))

                        r2 = float(
                            r2_score(
                                y_test,
                                preds,
                            )
                        )

                        evaluation_span.set_inputs(
                            {
                                "test_rows": len(y_test),
                            }
                        )

                        evaluation_span.set_outputs(
                            {
                                "mse": mse,
                                "rmse": rmse,
                                "r2": r2,
                            }
                        )

                    # -----------------------------------------
                    # Explicit metrics
                    # -----------------------------------------

                    mlflow.log_metrics(
                        {
                            "test_mse": mse,
                            "test_rmse": rmse,
                            "test_r2": r2,
                        }
                    )

                    print(f"  RMSE={rmse:.4f} R2={r2:.4f}")

                    # -----------------------------------------
                    # Best trial
                    # -----------------------------------------

                    if rmse < best_rmse:
                        best_rmse = rmse

                        best_trial = {
                            "trial_number": trial_number,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "random_state": args.random_state,
                            "mse": mse,
                            "rmse": rmse,
                            "r2": r2,
                            "run_id": child_run.info.run_id,
                        }

                        best_model = model

            # -------------------------------------------------
            # End pipeline trace
            # -------------------------------------------------

            if best_trial is not None:
                pipeline_span.set_outputs(
                    {
                        "best_trial": best_trial["trial_number"],
                        "best_rmse": best_trial["rmse"],
                        "best_r2": best_trial["r2"],
                    }
                )

        # -----------------------------------------------------
        # Parent run best metrics
        # -----------------------------------------------------

        if best_trial is not None:
            mlflow.log_params(
                {
                    "best_n_estimators": best_trial["n_estimators"],
                    "best_max_depth": best_trial["max_depth"],
                    "best_trial_number": best_trial["trial_number"],
                    "best_source_run_id": best_trial["run_id"],
                }
            )

            mlflow.log_metrics(
                {
                    "best_test_mse": best_trial["mse"],
                    "best_test_rmse": best_trial["rmse"],
                    "best_test_r2": best_trial["r2"],
                }
            )

        # -----------------------------------------------------
        # Logged Model management
        # -----------------------------------------------------
        #
        # Disabled until artifact storage is configured.
        #
        # Enable later:
        #
        # export MLFLOW_ENABLE_LOGGED_MODEL=true
        #

        if enable_logged_model and best_model is not None and best_trial is not None:
            manage_best_logged_model(
                best_model=best_model,
                best_trial=best_trial,
                X_train=X_train,
                experiment_name=args.experiment,
            )

        # -----------------------------------------------------
        # Get current artifact URI again
        # -----------------------------------------------------

        current_artifact_uri = mlflow.get_artifact_uri()

        mlflow.set_tag(
            "final_artifact_uri",
            current_artifact_uri,
        )

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        print("\n========== BEST TRIAL ==========")

        if best_trial is not None:
            print(f"Trial        : {best_trial['trial_number']}")

            print(f"n_estimators : {best_trial['n_estimators']}")

            print(f"max_depth    : {best_trial['max_depth']}")

            print(f"MSE          : {best_trial['mse']:.4f}")

            print(f"RMSE         : {best_trial['rmse']:.4f}")

            print(f"R2           : {best_trial['r2']:.4f}")

            print(f"Source Run ID: {best_trial['run_id']}")

        print("\n========== MLFLOW ==========")

        print(f"Parent Run ID: {parent_run.info.run_id}")

        print(f"Run Name     : {run_name}")

        print(f"Experiment   : {args.experiment}")

        print(f"Tracking URI : {tracking_uri}")

        print(f"Artifact URI : {current_artifact_uri}")

        print("\nTraining and MLflow tracking completed successfully.")

    # ---------------------------------------------------------
    # Search previous logged models
    # ---------------------------------------------------------

    print_logged_models(args.experiment)


if __name__ == "__main__":
    main()
