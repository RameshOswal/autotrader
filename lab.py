from data_load import DataPreprocess
from models import ARIMAUnivariate
from get_metrics import get_metrics
import numpy as np

if __name__=='__main__':

    train_dates = [['2014-07-01', '2016-05-07'], ['2014-11-01', '2016-09-07'], ['2015-02-01', '2016-12-08'],
                   ['2015-05-01', '2017-03-07']]
    test_dates = [['2016-05-07', '2016-06-27'], ['2016-09-07', '2016-10-28'], ['2016-12-08', '2017-01-28'],
                  ['2017-03-07', '2017-04-27']]

    input_folder_name = '../dataset/updated_poloniex_data'
    output_folder_name = '../dataset/Poloneix_Preprocessed'

    data = DataPreprocess(input_folder_name=input_folder_name,
                          output_folder_name=output_folder_name,
                          train_dates=train_dates,
                          test_dates=test_dates)
    data.asset_name()
    df = data.back_fwd_fill()
    ts_data = 100*df['train_2014-07-01_2016-05-07']['BTC_XMR_close'].values
    model = ARIMAUnivariate(1, 1, 0, 50)
    ytrue, ypred = model.evaluate(ts_data, 50)
    print(ytrue)
    print(ypred)
    assert (len(ytrue) == len(ypred))

    wts = ytrue[:-1]/ypred[1:]
    print(wts)
    m = get_metrics()
    m.apv_single_asset(ytrue[1:]/1000, wts, get_graph=True)
