import logging

logger = logging.getLogger(__name__)

class CurrentPricePredictor:
    """
    Price predictor using current price as prediction.
    
    Performance analysis for 30-day window, 10-step predictions:
    - Test MAE: ~110.47 EUR (0.12% relative)
    - Train MAE: ~134.41 EUR (0.14% relative)
    - Price volatility (CV): 1.27%
    
    Recommended feature adjustments for 30-day window:
    - Reduce return periods to: 1, 2, 3, 5, 8 (removed 13)
    - Reduce volatility windows to: 5, 10 (removed 20)
    - Reduce trend strength periods to: 5, 10 (removed 20, 30)
    - Reduce MA periods to: 5, 10 (removed 20)
    """
    
    def __init__(self):
        self.model = None
        self.test_mae = None
        self.train_mae = None
        self.price_stats = None
        self.volatility_threshold = 5.0
        self.forecast_steps = None
        self.window_size = None
        
    def fit(self, X, y, X_test=None, y_test=None, forecast_steps=10, window_size=30, **kwargs):
        """Calculate and store baseline performance metrics"""
        self.forecast_steps = forecast_steps
        self.window_size = window_size
        
        # Validate window size vs feature periods
        max_feature_period = max([
            col.split('_')[-1] for col in X.columns 
            if any(x in col for x in ['return_', 'volatility_', 'ma_', 'strength_'])
            and col.split('_')[-1].isdigit()
        ], default=0)
        
        if int(max_feature_period) > window_size / 3:
            logger.warning(
                f"Some feature periods ({max_feature_period}) are too long "
                f"relative to window size ({window_size}). Consider reducing periods."
            )
        
        current_prices = X['close'].values
        self.train_mae = abs(y - current_prices).mean()
        
        if X_test is not None and y_test is not None:
            test_prices = X_test['close'].values
            self.test_mae = abs(y_test - test_prices).mean()
        
        price_stats = {
            'mean': X['close'].mean(),
            'std': X['close'].std(),
            'cv': (X['close'].std() / X['close'].mean()) * 100,
            'train_mae_pct': (self.train_mae / X['close'].mean()) * 100,
            'test_mae_pct': (self.test_mae / X['close'].mean()) * 100 if self.test_mae else None,
            'n_samples': len(X),
            'forecast_steps': forecast_steps,
            'window_size': window_size
        }
        self.price_stats = price_stats
        
        logger.info("\nBaseline Model Performance:")
        logger.info(f"Training MAE: {self.train_mae:.2f} EUR ({price_stats['train_mae_pct']:.2f}%)")
        if self.test_mae:
            logger.info(f"Test MAE: {self.test_mae:.2f} EUR ({price_stats['test_mae_pct']:.2f}%)")
        logger.info(f"Price Mean: {price_stats['mean']:.2f} EUR")
        logger.info(f"Price Std: {price_stats['std']:.2f} EUR")
        logger.info(f"Price Volatility (CV): {price_stats['cv']:.2f}%")
        logger.info(f"Window Size: {window_size} days")
        logger.info(f"Forecast Steps: {forecast_steps}")
        
        if price_stats['cv'] > self.volatility_threshold:
            logger.warning(f"High price volatility detected: {price_stats['cv']:.2f}% > {self.volatility_threshold}%")
        
        if self.test_mae and self.test_mae > 1.5 * self.train_mae:
            logger.warning("Test MAE significantly higher than training MAE")
            
        logger.info("Using current price as optimal predictor")
        self.model = True

    def predict(self, X):
        """Return current price as prediction"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
            
        current_cv = (X['close'].std() / X['close'].mean()) * 100
        if current_cv > self.price_stats['cv'] * 1.5:
            logger.warning(f"Current volatility ({current_cv:.2f}%) significantly higher than training volatility ({self.price_stats['cv']:.2f}%)")
            
        return X['close'].values

    def get_model_obj(self):
        """Return the model object for saving"""
        return {
            'model': self.model,
            'price_stats': self.price_stats,
            'train_mae': self.train_mae,
            'test_mae': self.test_mae,
            'volatility_threshold': self.volatility_threshold,
            'forecast_steps': self.forecast_steps,
            'window_size': self.window_size
        }