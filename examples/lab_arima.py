from data_load.load_crypto import DataPreprocess
from models.arima_model import ARIMAModel
from get_metrics import get_metrics
from literals import ASSET_LIST

if __name__=='__main__':
    dp = DataPreprocess()
    for train_iter, test_iter, train_date, test_date in dp.load_train_test(asset_name=ASSET_LIST,feature_type='close'):
        # print(train_date, test_date)
        ts_data = 1000 * train_iter.values
        model = ARIMAModel(1, 1, 0, -1, 50)
        ytrue, ypred = model.evaluate(ts_data[:100], 50)
        weights = model.compute_allocation_weights(ypred, ytrue)
        assert (len(ytrue) == len(ypred))
        true_change = ytrue[1:]/ytrue[:-1]
        m = get_metrics(dt_range = test_date)
        m.apv_multiple_asset(true_change, weights, get_graph=True)
