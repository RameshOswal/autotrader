from data_load.load_crypto import DataPreprocess
from models.arima_model import ARIMAModel
from get_metrics import get_metrics
from literals import ASSET_LIST
import numpy as np

NUM_IDXS = 4
BPTT = 50
INIT_PV=1000
START_PT = 50
CONST_MULT=1000


if __name__=='__main__':
    dp = DataPreprocess(input_folder_name="../dataset/Poloneix_Preprocessednew")
    for idx in range(NUM_IDXS):
        train_df, test_df = dp.load_train_test(idx=idx,
                                                   path="../dataset/Poloneix_Preprocessednew",
                                                   asset_name=ASSET_LIST,feature_type='close')
        test_data = CONST_MULT*np.concatenate((train_df.values[-50:,:], test_df.values))
        test_date = "_".join(dp.test_dates[idx])
        model = ARIMAModel(1, 1, 0, -1, BPTT)
        ytrue, ypred = model.evaluate(test_data, START_PT)
        weights = model.compute_allocation_weights(ypred, ytrue)
        assert (len(ytrue) == len(ypred))
        true_change = ytrue[1:]/ytrue[:-1]
        m = get_metrics(dt_range = test_date)
        m.apv_multiple_asset(true_change, weights, get_graph=True, pv_0=INIT_PV)
