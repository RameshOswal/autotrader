import os
def fileInfo(pathname="dataset/poloneix_data\\BTC_BTCD.csv-2014-07-01 00_00_00-2016-05-07 00_00_00"):
    # arr = pathname.split('\\')[1].split(".csv-")
    # filename = arr[0]
    # start_date, end_date = arr[1][:19],arr[1][20:]
    # return filename, start_date, end_date
    basename = os.path.basename(pathname)
    a, b, start_date, end_date = basename.split('_')
    filename = a + '_' + b
    return filename, start_date, end_date

from glob import glob
import pandas as pd
import os

def save_dataset_files(dataset, basepath="", name_prefix=""):
    for file in dataset:
        filename = name_prefix+'_'+file+'.csv'
        print("**************** Saving ", filename, "*************************")
        train_df = dataset[file]
        train_df.to_csv(
            path_or_buf=os.path.join(basepath, filename),
            sep=",", header=True)
class DataPreprocess:

    def __init__(self,
                 input_folder_name = '../dataset/poloneix_data',
                 output_folder_name = '../dataset/poloneix_data/Poloneix_Preprocessed',
                 train_dates=[[]],
                 test_dates=[[]]):
        self.input_folder_name = input_folder_name
        self.output_folder_name = output_folder_name
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.assets = []
        self.dataset = {}

        self.load_dataset()

    def load_dataset(self):
        print("*******************Trying to load old files:")
        file_name_list = []
        for start,end in self.train_dates:
            for path in  glob(os.path.join(self.output_folder_name, "*"+start+"*"+end+"*")):
                name = os.path.basename(path)
                self.dataset['train_'+start+'_'+end] = pd.read_csv(path, header=0, index_col=0)
                file_name_list += [name]
        for start,end in self.test_dates:
            for path in  glob(os.path.join(self.output_folder_name, "*"+start+"*"+end+"*")):
                name = os.path.basename(path)
                self.dataset['test_'+start+'_'+end] = pd.read_csv(path, header=0, index_col=0)
                file_name_list += [name]
        if len(file_name_list) != 0:
            print("******************* Following files were found to be present and loaded in dataset:" + "\n\t".join(file_name_list))

    def dropna(self):
        newdataset = {}
        for file in self.dataset:
            newdataset[file] = self.dataset[file].dropna(axis=0)
        return newdataset

    def back_fwd_fill(self):
        newdataset = {}
        for file in self.dataset:
            newdataset[file] = self.dataset[file].fillna(method='bfill').fillna(method='ffill')
        return newdataset

    def asset_name(self):
        self.assets = set()
        for path in glob(os.path.join(self.input_folder_name, 'train', "*")) + glob(os.path.join(self.input_folder_name, 'test',"*")):
            filename, _, _ = fileInfo(path)
            self.assets.add(filename)

    def preprocess(self, dates=[], file='train', path_postfix=""):
        print("********************Preprocessing: ", file)
        for start_date, end_date in dates:
            train_df = pd.DataFrame()
            for asset in self.assets:
                list_files = glob(os.path.join(self.input_folder_name , path_postfix,"*" + asset + "*" + start_date + "*" + end_date) + "*" )
                if len(list_files) != 0:
                    for path in list_files:
                        file_name, _, _ = fileInfo(path)
                        df = pd.read_csv(path, header=0)
                        df.columns = [file_name + '_' + str(i) if i !='date' else 'date' for i in df.columns]
                        # train_df = pd.concat((train_df , df), axis=1) if train_df.shape[0] else df
                        train_df = pd.merge(train_df, df, how='outer', on='date') if train_df.shape[0] else df
                else :
                    print("************Warning: Data Missing for ", asset, " between ", start_date, " to ", end_date, "**************" )
            print("********************************Saving File :"" between ", start_date, " to ", end_date, ' ***************************' )
            self.dataset[file+'_'+start_date+'_'+end_date] = train_df

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
    save_dataset_files(df, basepath=output_folder_name, name_prefix="bffill")
    # data.preprocess(dates=data.train_dates, file='train', path_postfix="train")
    # data.preprocess(dates=data.test_dates, file='test', path_postfix="test")
