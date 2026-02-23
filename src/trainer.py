"""
VehiclePriceTrainer
-------------------
A self-contained class for training, evaluating, and persisting
vehicle sale-price prediction models with MLflow experiment tracking.
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Columns explicitly banned by the brief
BANNED_FEATURES = [
    "AvgWholesale", "AvgRetail", "GoodWholesale", "GoodRetail",
    "TradeMin", "TradeMax", "PrivateMax",
]

TARGET = "Sold_Amount"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict of standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


class VehiclePriceTrainer:
    """
    Train a vehicle sale-price regression model.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fully-specified sklearn Pipeline ending with a regressor.
    experiment_name : str
        MLflow experiment name used for all runs.
    cv_folds : int
        Number of cross-validation folds (default 5).
    models_dir : str | Path
        Directory where trained models are persisted.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        experiment_name: str = "vehicle_price",
        cv_folds: int = 5,
        models_dir: str | Path = "models",
    ):
        self.pipeline = pipeline
        self.experiment_name = experiment_name
        self.cv_folds = cv_folds
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        run_name: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> "VehiclePriceTrainer":
        """
        Fit the pipeline on training data and log everything to MLflow.

        Parameters
        ----------
        X_train, y_train : training features and target.
        X_val, y_val : optional held-out validation set.
        run_name : optional MLflow run label.
        extra_tags : optional key-value tags stored on the run.
        """
        self._validate_features(X_train)
        run_name = run_name or self._default_run_name()

        with mlflow.start_run(run_name=run_name) as run:
            self._log_tags(extra_tags)
            self._log_params()

            # ---------- cross-validation ----------
            logger.info("Starting %d-fold cross-validation …", self.cv_folds)
            cv_start = time.time()
            cv_scores = self._cross_validate(X_train, y_train)
            cv_elapsed = time.time() - cv_start
            self._log_cv_metrics(cv_scores, cv_elapsed)

            # ---------- final fit on full training set ----------
            logger.info("Fitting on full training set …")
            train_start = time.time()
            self.pipeline.fit(X_train, y_train)
            train_elapsed = time.time() - train_start
            mlflow.log_metric("train_time_s", round(train_elapsed, 2))

            # ---------- training-set metrics ----------
            train_preds = self.pipeline.predict(X_train)
            train_metrics = regression_metrics(y_train.values, train_preds)
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", round(v, 4))

            # ---------- validation metrics (if provided) ----------
            if X_val is not None and y_val is not None:
                val_preds = self.pipeline.predict(X_val)
                val_metrics = regression_metrics(y_val.values, val_preds)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", round(v, 4))
                logger.info(
                    "Val  MAE=%.2f  RMSE=%.2f  R2=%.4f  MAPE=%.4f",
                    val_metrics["MAE"], val_metrics["RMSE"],
                    val_metrics["R2"], val_metrics["MAPE"],
                )

            # ---------- persist model ----------
            model_path = self.models_dir / f"{run_name}.pkl"
            joblib.dump(self.pipeline, model_path)
            mlflow.log_artifact(str(model_path))
            mlflow.sklearn.log_model(self.pipeline, "model")

            self.run_id_ = run.info.run_id
            logger.info("Run %s finished.  Model saved to %s.", run.info.run_id, model_path)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (requires fit to have been called)."""
        self._check_fitted()
        self._validate_features(X)
        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return regression metrics on a labelled dataset."""
        preds = self.predict(X)
        metrics = regression_metrics(y.values, preds)
        for k, v in metrics.items():
            logger.info("  %-6s %.4f", k, v)
        return metrics

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the fitted pipeline with joblib."""
        self._check_fitted()
        path = Path(path) if path else self.models_dir / "latest_model.pkl"
        joblib.dump(self.pipeline, path)
        logger.info("Pipeline saved to %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "VehiclePriceTrainer":
        """Load a previously saved pipeline (returns a bare wrapper)."""
        path = Path(path)
        trainer = VehiclePriceTrainer.__new__(VehiclePriceTrainer)
        trainer.pipeline = joblib.load(path)
        return trainer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scoring_map = {
            "neg_mae": "neg_mean_absolute_error",
            "neg_rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        results = {}
        for label, scorer in scoring_map.items():
            scores = cross_val_score(
                self.pipeline, X, y, cv=kf, scoring=scorer, n_jobs=-1
            )
            results[label] = scores
        return results

    def _log_cv_metrics(self, cv_scores: dict, elapsed: float) -> None:
        mae_scores = -cv_scores["neg_mae"]
        rmse_scores = -cv_scores["neg_rmse"]
        r2_scores = cv_scores["r2"]

        mlflow.log_metric("cv_mae_mean", round(mae_scores.mean(), 4))
        mlflow.log_metric("cv_mae_std", round(mae_scores.std(), 4))
        mlflow.log_metric("cv_rmse_mean", round(rmse_scores.mean(), 4))
        mlflow.log_metric("cv_rmse_std", round(rmse_scores.std(), 4))
        mlflow.log_metric("cv_r2_mean", round(r2_scores.mean(), 4))
        mlflow.log_metric("cv_r2_std", round(r2_scores.std(), 4))
        mlflow.log_metric("cv_time_s", round(elapsed, 2))

        logger.info(
            "CV  MAE=%.2f±%.2f  RMSE=%.2f±%.2f  R2=%.4f±%.4f",
            mae_scores.mean(), mae_scores.std(),
            rmse_scores.mean(), rmse_scores.std(),
            r2_scores.mean(), r2_scores.std(),
        )

    def _log_params(self) -> None:
        """Log pipeline step names and (shallow) estimator params."""
        for step_name, estimator in self.pipeline.steps:
            if hasattr(estimator, "get_params"):
                params = estimator.get_params(deep=False)
                for k, v in params.items():
                    try:
                        mlflow.log_param(f"{step_name}__{k}", v)
                    except Exception:
                        pass  # skip un-serialisable params silently
        mlflow.log_param("cv_folds", self.cv_folds)

    def _log_tags(self, extra: dict[str, str] | None) -> None:
        mlflow.set_tag("framework", "sklearn")
        if extra:
            for k, v in extra.items():
                mlflow.set_tag(k, v)

    def _validate_features(self, X: pd.DataFrame) -> None:
        leaks = [c for c in BANNED_FEATURES if c in X.columns]
        if leaks:
            raise ValueError(
                f"Banned feature(s) present in input: {leaks}. "
                "Remove them before training."
            )

    def _check_fitted(self) -> None:
        if not hasattr(self.pipeline, "named_steps"):
            raise RuntimeError("Pipeline not fitted yet. Call .fit() first.")
        last_step = list(self.pipeline.named_steps.values())[-1]
        if not hasattr(last_step, "feature_importances_") and not hasattr(
            last_step, "coef_"
        ):
            pass  # some estimators don't expose these; skip check

    def _default_run_name(self) -> str:
        estimator = list(self.pipeline.named_steps.values())[-1]
        return f"{type(estimator).__name__}_{int(time.time())}"
