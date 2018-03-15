from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


class ARIMAUnivariate:
    def __init__(self, ar_order, num_diffs, ma_order, bptt=None):
        """
        Univariate ARIMA Model - Predicts a single time series using
        the history of that same series
        ar_order: Lag order for the AR model
        num_diffs: Level of differencing
        ma_order: Lag order for the MA model
        """
        self.order = (ar_order, num_diffs, ma_order)
        self.bptt = bptt

    def evaluate(self, endog_data, start_pt):
        if self.bptt is not None:
            assert (start_pt >= self.bptt)
        predicted_forecast = []
        true_forecast = endog_data[start_pt:]
        num_samples = len(endog_data)
        for t in range(start_pt, num_samples):
            endog_history = (endog_data[:t] if self.bptt is None else endog_data[t - self.bptt:t])
            model = ARIMA(endog_history, order=self.order).fit(disp=0)
            predicted_forecast = np.append(predicted_forecast, model.forecast()[0])
        return true_forecast, predicted_forecast

    def forecast(self, endog_history, steps_ahead):
        model = ARIMA(endog_history[-self.bptt:], order=self.order).fit(disp=0)
        return model.forecast(steps=steps_ahead)[0]