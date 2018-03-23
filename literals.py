FEATURE_TYPE = ['high',
                'low',
                'open',
                'quoteVolume',
                'volume',
                'weightedAverage']
                
STOCK_FEATURE_TYPE = ['Open',
					  'High',
					  'Low',
					  'Close',
					  'Volumne']

CRYPTO_TRAIN_DATES = [['2014-07-01', '2016-05-07'], ['2014-11-01', '2016-09-07'], ['2015-02-01', '2016-12-08'],
                              ['2015-05-01', '2017-03-07']]
CRYPTO_TEST_DATES = [['2016-05-07', '2016-06-27'], ['2016-09-07', '2016-10-28'], ['2016-12-08', '2017-01-28'],
                      ['2017-03-07', '2017-04-27']]

ASSET_LIST = ['BTC_BTCD', 'BTC_DASH', 'BTC_ETH',
              'BTC_FCT', 'BTC_LTC', 'BTC_XEM',
              'BTC_XMR', 'BTC_XRP', 'USDT_BTC']

CRYPTO_INPUT_FOLDER = 'dataset/updated_poloniex_data'
CRYPTO_OUTPUT_FOLDER = 'dataset/Poloneix_Preprocessed'