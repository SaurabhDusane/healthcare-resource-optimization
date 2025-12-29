"""Time Series Forecasting Module"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """Time series forecasting models."""
    
    def __init__(self):
        self.logger = logger
        self.model = None
        self.forecast = None
    
    def prepare_data(self, df, target_col, date_col):
        """Prepare time series data."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        return df[target_col]
    
    def train_prophet(self, df, target_col='visits', date_col='date'):
        """Train Prophet model."""
        try:
            from prophet import Prophet
            
            self.logger.info("Training Prophet model...")
            
            df_prophet = pd.DataFrame({
                'ds': df[date_col],
                'y': df[target_col]
            })
            
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            self.model.fit(df_prophet)
            self.logger.info("Prophet model trained")
            
            return self.model
            
        except ImportError:
            self.logger.error("Prophet not installed")
            return None
    
    def forecast_prophet(self, periods=30):
        """Generate forecast using Prophet."""
        if self.model is None:
            self.logger.error("Model not trained")
            return None
        
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
        
        return self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def evaluate_forecast(self, y_true, y_pred):
        """Evaluate forecast accuracy."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Accuracy': 100 - mape
        }
