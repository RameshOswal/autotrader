from data_load import DataPreprocess
from models import ARIMAModel
from get_metrics import get_metrics
import numpy as np

if __name__=='__main__':

    dp = DataPreprocess()
    for train_iter, test_iter in dp.load_train_test():
        ts_data = 100*train_iter.values
        model = ARIMAModel(1, 1, 0, 50)
        ytrue, ypred = model.evaluate(ts_data[:200], 50)
        assert (len(ytrue) == len(ypred))

        wts = ytrue[:-1]/ypred[1:]
        m = get_metrics()
        m.apv_single_asset(ytrue[1:]/1000, wts, get_graph=True)
        exit(0)
