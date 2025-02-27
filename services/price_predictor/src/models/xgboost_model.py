from typing import Optional

from loguru import logger
import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

class XGBoostModel:

    def __init__(self):
        self.model: XGBRegressor = None
        self.best_params = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_search_trials: Optional[int] = 5,
        n_splits: Optional[int] = 2,
    ):
        """
        Trains an XGBoost model on the given training data.
        """
        logger.info(f"Training XGBoost model with n_search_trials={n_search_trials} and n_splits={n_splits}")
        
        try:
            if n_search_trials > 0:
                study = optuna.create_study(direction="minimize")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                        'tree_method': 'hist',
                        'enable_categorical': False
                    }
                    
                    # Use TimeSeriesSplit for validation
                    tscv = TimeSeriesSplit(n_splits=n_splits)
                    scores = []
                    
                    for train_idx, val_idx in tscv.split(X_train):
                        X_fold_train = X_train.iloc[train_idx]
                        y_fold_train = y_train.iloc[train_idx]
                        X_fold_val = X_train.iloc[val_idx]
                        y_fold_val = y_train.iloc[val_idx]
                        
                        model = XGBRegressor(**params)
                        model.fit(X_fold_train, y_fold_train)
                        
                        y_pred = model.predict(X_fold_val)
                        mae = mean_absolute_error(y_fold_val, y_pred)
                        scores.append(mae)
                    
                    return sum(scores) / len(scores)
                
                study.optimize(objective, n_trials=n_search_trials, timeout=300)
                self.best_params = study.best_params
                self.best_params.update({
                    'tree_method': 'hist',
                    'enable_categorical': False
                })
                logger.info(f"Best parameters: {self.best_params}")
                
                # Train final model with best parameters
                self.model = XGBRegressor(**self.best_params)
                self.model.fit(X_train, y_train)
                logger.info("Final model trained with best parameters")

            else:
                # Use default parameters
                params = {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'tree_method': 'hist',
                    'enable_categorical': False
                }
                self.model = XGBRegressor(**params)
                self.model.fit(X_train, y_train)

            return self

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            # Use default parameters if optimization fails
            params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',
                'enable_categorical': False
            }
            self.model = XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            logger.info("Fallback to default parameters due to error")
            
            return self
    
    def predict(self, X_test):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X_test)
    

    def get_model_obj(self):
        return self.model