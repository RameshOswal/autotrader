from data_scraper import *
import pandas as pd
import datetime, time
def totimestamp(dt, epoch=datetime.datetime(1970,1,1)):
	td = dt - epoch
	return (td.microseconds+(td.seconds+td.days*86400)*10**6) / 10**6

def get_streaming_crypto(asset_name=['BTC_BTCD', 'BTC_DASH', 'BTC_ETH',
              			'BTC_FCT', 'BTC_LTC', 'BTC_XEM',
              			'BTC_XMR', 'BTC_XRP', 'USDT_BTC'],
              			feature_type='open', history=50):
	while(True):
		end_time = totimestamp(datetime.datetime.now())
		start_time = end_time-100000
		dataset = pd.DataFrame()
		if type(asset_name) == str:
			asset_name = [asset_name]
		for asset in asset_name:
			df = get_data(asset, start_time, end_time, mode="test", return_df=True)
			df = df[['date', feature_type]]
			df.columns = [ i if i=='date' else asset +'_'+i for i in df.columns ]
			dataset = dataset.merge(df, on='date', how="inner") if dataset.size != 0 else df
		dataset = dataset.iloc[-1*history:]

		yield dataset

		while( (end_time+300) > totimestamp(datetime.datetime.now())):
			time.sleep(2)
			pass #just wait till 50 seconds pass

if __name__=="__main__":
	for data in get_streaming_crypto(asset_name='BTC_BTCD'):
		print(data.shape)