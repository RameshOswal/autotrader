from data_load import DataPreprocess
from models import ARIMAModel
from get_metrics import get_metrics
import numpy as np

if __name__=='__main__':
    dp = DataPreprocess()
    for train_iter, test_iter in dp.load_train_test(asset_name=["BTC_XEM", "BTC_LTC"],feature_type='close'):
        ts_data = 1000 * train_iter.values
        model = ARIMAModel(1, 1, 0, 100)
        ytrue, ypred = model.evaluate(ts_data[:5000], 100)
        weights = model.compute_allocation_weights(ypred, ytrue)
        assert (len(ytrue) == len(ypred))
        true_change = ytrue[1:]/ytrue[:-1]
        m = get_metrics()
        m.apv_multiple_asset(true_change, weights, get_graph=True)
        exit(0)
