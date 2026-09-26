from __future__ import annotations

import argparse
import math
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple MLflow demo - Wine Quality Prediction"
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
        "--run",
        default="run-2",
        help="MLflow run name (default: run-2)",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="RandomForest n_estimators (default: 50)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="RandomForest max_depth (default: 5)",
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


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # MLflow configuration
    # ---------------------------------------------------------
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:7006",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    print(f"MLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {args.experiment}")
    print(f"CSV                 : {args.csv}")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}. "
            "Create or copy wine_sample.csv into the data directory."
        )

    df = pd.read_csv(args.csv)

    print(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # Validate target column
    # ---------------------------------------------------------
    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------
    # Prepare features and target
    # ---------------------------------------------------------
    #
    # Remove:
    #   - target column: quality
    #   - Id column: identifier, not a useful ML feature
    #
    columns_to_drop = [args.target]

    if "Id" in df.columns:
        columns_to_drop.append("Id")

    X = df.drop(columns=columns_to_drop)
    y = df[args.target]

    print(f"Features: {list(X.columns)}")
    print(f"Target: {args.target}")
    print(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # Train/test split
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
    # Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run(run_name=args.run) as run:
        # -----------------------------------------------------
        # Log parameters
        # -----------------------------------------------------
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("target", args.target)

        # -----------------------------------------------------
        # Create model
        # -----------------------------------------------------
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------
        print("\nTraining model...")

        model.fit(X_train, y_train)

        print("Training completed.")

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------
        preds = model.predict(X_test)

        # -----------------------------------------------------
        # Calculate metrics
        # -----------------------------------------------------
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        # -----------------------------------------------------
        # Log metrics
        # -----------------------------------------------------
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # -----------------------------------------------------
        # Log trained model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        # -----------------------------------------------------
        # Print results
        # -----------------------------------------------------
        print("\n========== RESULTS ==========")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        print("\n========== MLFLOW ==========")
        print(f"Run ID    : {run.info.run_id}")
        print(f"Experiment: {args.experiment}")
        print(f"Tracking  : {tracking_uri}")

        print("\nModel logged successfully.")


if __name__ == "__main__":
    main()
