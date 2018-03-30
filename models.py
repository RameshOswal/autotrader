from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from joblib import Parallel, delayed
import warnings

def _evaluate_single(arg, **kwarg):
    return ARIMAModel.evaluate_single(*arg, **kwarg)


class ARIMAModel:
    def __init__(self, ar_order, num_diffs, ma_order, num_jobs=-1, bptt=None):
        """
        ar_order: Lag order for the AR model
        num_diffs: Level of differencing
        ma_order: Lag order for the MA model
        """
        self.order = (ar_order, num_diffs, ma_order)
        self.bptt = bptt
        self.num_jobs = num_jobs

    def evaluate(self, endog_data, start_pt):
        num_dims = len(endog_data.shape)
        assert 0 < num_dims <= 2
        if num_dims == 1 or endog_data.shape[1] == 1:
            endog_data = endog_data.flatten()
            return self.evaluate_single(endog_data, start_pt)
        else:
            return self.parallel_evaluate_single(endog_data, start_pt, endog_data.shape[1])

    def forecast(self, endog_history, steps_ahead):
        num_dims = len(endog_history.shape)
        assert 0 < num_dims <= 2
        if endog_history.shape[1] == 1 or num_dims == 1:
            endog_history = endog_history.flatten()
            return self.forecast_single(endog_history, steps_ahead)
        else:
            results = [self.forecast_single(endog_history[:, i], steps_ahead) for i in range(endog_history.shape[1])]
            return np.vstack(results)

    def parallel_evaluate_single(self, endog_data, start_pt, num_stocks):
        results = Parallel(n_jobs=self.num_jobs, backend="threading",verbose=4) \
            (delayed(_evaluate_single)(i) for i in zip([self] * num_stocks,
                                                       [endog_data[:, s] for s in range(num_stocks)],
                                                       [start_pt]*num_stocks))
        ytrue_stacked = [results[i][0] for i in range(num_stocks)]
        ypred_stacked = [results[i][1] for i in range(num_stocks)]
        return np.vstack(ytrue_stacked).T, np.vstack(ypred_stacked).T

    def evaluate_single(self, endog_data, start_pt):
        if self.bptt is not None:
            assert (start_pt >= self.bptt)
        predicted_forecast = []
        num_samples = len(endog_data)
        for t in range(start_pt, num_samples):
            endog_history = (endog_data[:t] if self.bptt is None else endog_data[t - self.bptt:t])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
