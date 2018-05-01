__author__ = 'deeptrader'

import os
import time
import pandas as pd
import datetime
"""
Alpha vantage key = FMMKEQFPENW58OEI

"""

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=1800"
#PAIR_LIST = ["BTC_ETH"]
DATA_DIR = "data/stock"
COLUMNS = ["date","high","low","open","close","volume","quoteVolume","weightedAverage"]

def get_data(pair, start, end, mode = "train", return_df=False):
    datafile = os.path.join(DATA_DIR + "/{}".format(mode), pair)
    timefile = os.path.join(DATA_DIR + "/{}".format(mode), pair)

    # if os.path.exists(datafile):
    #     newfile = False
    #     start_time = int(open(timefile).readline()) + 1
    # else:
    #     newfile = True
    start_time = start
    end_time = end
    # print( pair, start_time, end_time)
    # exit()
    url = FETCH_URL % (pair, start_time, end_time)
    print("Get %s from %d to %d" % (pair, start_time, end_time))

    df = pd.read_json(url, convert_dates = True)
    if return_df == True:
        return df
    #import pdb;pdb.set_trace()

    if df["date"].iloc[-1] == 0:
        print("No data.")
        with open("no_data_logs.txt", "a") as flp:
            flp.write("{},{},{}".format(pair, str(datetime.datetime.fromtimestamp(start_time)).split(" ")[0], str(datetime.datetime.fromtimestamp(end_time)).split(" ")[0] + "\n"))
        return

    # end_time = df["date"].iloc[-1]
    #ft = open(timefile,"w")
    #ft.write("%d\n" % end_time)
    # ft.close()
    outf = open('{}_{}_{}.csv'.format(datafile, str(datetime.datetime.fromtimestamp(start_time)).split(" ")[0], str(datetime.datetime.fromtimestamp(end_time)).split(" ")[0]), "a")
    df.to_csv(outf, index=False, header=True)
    outf.close()
    print("Finish.")
    # exit()
    time.sleep(3)


def main():
    with open("no_data_logs.txt", "a") as flp: flp.write("ticker,start_time,end_time" + "\n")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    df = pd.read_json("https://poloniex.com/public?command=return24hVolume")
    asset_list = ['BTC_BTCD', 'BTC_ETH', 'BTC_LTC', 'BTC_XRP', 'USDT_BTC', 'BTC_ETC', 'BTC_DASH', 'BTC_XMR', 'BTC_XEM', 'BTC_FCT', 'BTC_GNT', 'BTC_ZEC']
    # asset_list = ['AAPL', 'MSFT', 'FB', 'GOOGL', 'AMZN']
    # print(df.keys())
    # exit()
    pairs = [pair for pair in df.keys() if pair in asset_list ]
    # print(set(pairs).symmetric_difference(set(asset_list)))
    # print(pairs)
    # exit()
    # te_start_time, te_end_time = [int(datetime.datetime(2016,9,7).strftime('%s')), int(datetime.datetime(2016,12,8).strftime('%s')), int(datetime.datetime(2017,3,7).strftime('%s')), int(datetime.datetime(2016,5,7).strftime('%s'))], \
    #                              [int(datetime.datetime(2016,10,28).strftime('%s')), int(datetime.datetime(2017,1,28).strftime('%s')), int(datetime.datetime(2017,4,27).strftime('%s')), int(datetime.datetime(2016,6,27).strftime('%s'))]

    te_start_time, te_end_time = [int(datetime.datetime(2017, 3, 7).strftime('%s'))], \
                                 [int(datetime.datetime(2017, 4, 27).strftime('%s'))]

    tr_start_time, tr_end_time = [int(datetime.datetime(2014,11,1).strftime('%s')), int(datetime.datetime(2015,2,1).strftime('%s')), int(datetime.datetime(2015,5,1).strftime('%s')), int(datetime.datetime(2014,7,1).strftime('%s'))], \
                                 [int(datetime.datetime(2016,9,7).strftime('%s')), int(datetime.datetime(2016,12,8).strftime('%s')), int(datetime.datetime(2017,3,7).strftime('%s')), int(datetime.datetime(2016,5,7).strftime('%s'))]
    # exit()
    # for start, end in zip(tr_start_time, tr_end_time):
    #     print('Fetching training data from {} to {}'.format(datetime.datetime.fromtimestamp(start), datetime.datetime.fromtimestamp(end)))
    #     for pair in pairs:
    #         get_data(pair, start, end, mode = "train")
    #         time.sleep(2)
    for start, end in zip(te_start_time, te_end_time):
        print('Fetching testing data from {} to {}'.format(datetime.datetime.fromtimestamp(start), datetime.datetime.fromtimestamp(end)))
        for pair in pairs:
            get_data(pair, start, end, mode = "test")
            time.sleep(2)


def check_function(ticker = "MSFT"):
    import csv
    import datetime
    import re

    import pandas as pd
    import requests

    def get_google_finance_intraday(ticker, period=60, days=1):
        """
        Retrieve intraday stock data from Google Finance.
        Parameters
        ----------
        ticker : str
            Company ticker symbol.
        period : int
            Interval between stock values in seconds.
        days : int
            Number of days of data to retrieve.
        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the opening price, high price, low price,
            closing price, and volume. The index contains the times associated with
            the retrieved price values.
        """

        uri = 'http://www.google.com/finance/getprices' \
              '?i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker,
                                                                              period=period,
                                                                              days=days)
        page = requests.get(uri)
        # reader = list(map(lambda x: str(x, 'utf-8'), page.content.splitlines()))
        # exit()
        reader = csv.reader(map(lambda x : x.decode('utf-8'), page.content.splitlines()))
        # reader = reader.decode('utf-8')
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        rows = []
        times = []
        print(reader)
        for row in reader:
            # print(row)
            # exit()
            if re.match('^[a\d]', row[0]):
                if row[0].startswith('a'):
                    # print row
                    start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                    times.append(start)
                else:
                    times.append(start + datetime.timedelta(seconds=period * int(row[0])))
                rows.append(map(float, row[1:]))
        if len(rows):
            return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'),
                                columns=columns)
        else:
            return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

    df = get_google_finance_intraday(ticker=ticker, period=300, days=60)
    df.to_csv("{}_stock_price.csv".format(ticker))
    print(df.head(), len(df))

if __name__ == '__main__':
    stock_lst = ["MSFT", "AMZN", "FB", "AAPL", "GOOGL"]
    for tick in stock_lst: check_function(ticker=tick)
    # main()
