from loguru import logger
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from typing import Optional
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class XGBoostModel:
    """
    A wrapper around XGBoost that handles hyperparameter tuning and training.
    """
    def __init__(self):
        self.model = None
        self.best_params = None

    def objective(self, trial, X, y, n_splits):
        """Optuna objective function for hyperparameter tuning"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist'  # Use histogram-based algorithm for faster training
        }

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        try:
            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                model = XGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                mae = np.mean(np.abs(y_val_fold - y_pred))
                scores.append(mae)

            return np.mean(scores)
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return float('inf')  # Return a large value on error

    def fit(self, X, y, n_search_trials: Optional[int] = 10, n_splits: Optional[int] = 3):
        """
        Train the model with hyperparameter tuning if n_search_trials > 0
        """
        logger.info(f"Training the XGBoost model with {n_search_trials} search trials and {n_splits} splits")

        try:
            if n_search_trials > 0:
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    lambda trial: self.objective(trial, X, y, n_splits),
                    n_trials=n_search_trials,
                    catch=(Exception,)
                )
                self.best_params = study.best_params
                self.best_params.update({
                    'tree_method': 'hist'
                })
            else:
                self.best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'tree_method': 'hist'
                }

            logger.info(f"Best parameters: {self.best_params}")
            self.model = XGBRegressor(**self.best_params)
            self.model.fit(X, y)
            return self

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            # Use default parameters if optimization fails
            self.best_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist'
            }
            self.model = XGBRegressor(**self.best_params)
            self.model.fit(X, y)
            return self

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)

