def fileInfo(pathname="dataset/poloneix_data\\BTC_BTCD.csv-2014-07-01 00_00_00-2016-05-07 00_00_00"):
    arr = pathname.split('\\')[1].split(".csv-")
    filename = arr[0]
    start_date, end_date = arr[1][:19],arr[1][20:]
    return filename, start_date, end_date

from glob import glob
import pandas as pd
import os

class DataPreprocess:

    def __init__(self,
                 input_folder_name = '../dataset/poloneix_data',
                 output_folder_name = '../dataset/poloneix_data/Poloneix_Preprocessed',
                 train_dates=[['2014-07-01', '2016-05-07'], ['2014-11-01', '2016-09-07'], ['2015-02-01', '2016-12-08'], ['2015-05-01', '2017-03-07']],
                 test_dates=[['2016-05-07', '2016-06-27'], ['2016-09-07', '2016-10-28'], ['2016-12-08', '2017-01-28'], ['2017-03-07', '2017-04-27']]):
        self.input_folder_name = input_folder_name
        self.output_folder_name = output_folder_name
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.assets = []

    def asset_name(self):
        self.assets = set()
        for path in glob(os.path.join(self.input_folder_name , "*")):
            filename, _, _ = fileInfo(path)
            self.assets.add(filename)

    def preprocess(self):
        for start_date, end_date in self.train_dates:
            train_df = pd.DataFrame()
            for asset in self.assets:
                list_files = glob(os.path.join(self.input_folder_name , "*" + asset + "*" + start_date + "*" + end_date) + "*" )
                if len(list_files) != 0:
                    for path in list_files:
                        file_name, _, _ = fileInfo(path)
                        df = pd.read_csv(path, header=None)
                        df.columns = [file_name + str(i) if i !=1 else 'date' for i in df.columns]
                        # train_df = pd.concat((train_df , df), axis=1) if train_df.shape[0] else df
                        train_df = pd.merge(train_df, df, how='outer', on='date') if train_df.shape[0] else df
                else :
                    print("************Warning: Data Missing for ", asset, " between ", start_date, " to ", end_date, "**************" )
            print("********************************Saving File :"" between ", start_date, " to ", end_date, ' ***************************' )
            train_df.to_csv(path_or_buf=os.path.join(self.output_folder_name, 'train_'+start_date+'_to_'+end_date+'.csv'),
                            sep=",", header=True)
#         self.DataFrame = pd.DataFrame()
#         for path in glob(self.folder_name):
#             file_name, _, _ = fileInfo(path)
#             df = pd.read_csv(path, header=None)
#             df.columns = [file_name + str(i) for i in df.columns]
#             self.DataFrame = pd.concat((self.DataFrame, df), axis=1) if self.DataFrame.shape[0] else df


if __name__=='__main__':
    data = DataPreprocess(input_folder_name='../dataset/poloneix_data',
                          output_folder_name = '../dataset/Poloneix_Preprocessed')
    data.asset_name()
    print(len(data.assets), data.assets)
    data.preprocess()
    # print(data.DataFrame)