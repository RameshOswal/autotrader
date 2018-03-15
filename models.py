from statsmodels.tsa.arima_model import ARIMA
import numpy as np


class ARIMAModel:
    def __init__(self, ar_order, num_diffs, ma_order, bptt=None):
        """
        ar_order: Lag order for the AR model
        num_diffs: Level of differencing
        ma_order: Lag order for the MA model
        """
        self.order = (ar_order, num_diffs, ma_order)
        self.bptt = bptt

    def evaluate(self, endog_data, start_pt):
        num_dims = len(endog_data.shape)
        assert 0 < num_dims <= 2
        if num_dims == 1 or endog_data.shape[1] == 1:
            endog_data = endog_data.flatten()
            return self.evaluate_single(endog_data, start_pt)
        else:
            results_true = None
            results_predicted = None
            for i in range(endog_data.shape[1]):
                ytrue, ypred = self.evaluate_single(endog_data[:,i], start_pt)
                if results_true is None:
                    results_true = ytrue
                    results_predicted = ypred
                else:
                    results_true = np.vstack((results_true, ytrue))
                    results_predicted = np.vstack((results_predicted, ypred))
            return results_true.T, results_predicted.T

    def forecast(self, endog_history, steps_ahead):
        num_dims = len(endog_history.shape)
        assert 0 < num_dims <= 2
        if endog_history.shape[1] == 1 or num_dims == 1:
            endog_history = endog_history.flatten()
            return self.forecast_single(endog_history, steps_ahead)
        else:
            results = [self.forecast_single(endog_history[:, i], steps_ahead) for i in range(endog_history.shape[1])]
            return np.vstack(results)


    def evaluate_single(self, endog_data, start_pt):
        if self.bptt is not None:
            assert (start_pt >= self.bptt)
        predicted_forecast = []
        num_samples = len(endog_data)
        for t in range(start_pt, num_samples):
            endog_history = (endog_data[:t] if self.bptt is None else endog_data[t - self.bptt:t])
            model = ARIMA(endog_history, order=self.order).fit(disp=0)
            predicted_forecast = np.append(predicted_forecast, model.forecast()[0])
        true_forecast = endog_data[start_pt:]
        return true_forecast, predicted_forecast

    def forecast_single(self, endog_history, steps_ahead):
        model = ARIMA(endog_history[-self.bptt:], order=self.order).fit(disp=0)
        return model.forecast(steps=steps_ahead)[0]

    def normalize(self, price_weights):
        return price_weights/price_weights.sum()

    def compute_allocation_weights(self, ypred, ytrue, cash_reserve=0.01):
        assert len(ypred.shape) == 2, "Number of dimensions must be 2"
        num_timesteps, num_stocks = ypred.shape
        alloc_weights = np.zeros((num_timesteps, num_stocks + 1),dtype=np.float)
        alloc_weights[0][0] = 1.0
        for t in range(1, num_timesteps):
            rel_change = (ypred[t] - ytrue[t - 1])/ytrue[t - 1]
            rel_change += 1.0
            rel_change = rel_change/rel_change.sum()
            rel_change -= cash_reserve
            rel_change = np.maximum(rel_change, 0)
            alloc_weights[t][1:] = rel_change
            alloc_weights[t][0] = 1.0 - rel_change.sum()
        return alloc_weights
