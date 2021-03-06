import os
from data_load import helper
import numpy as np

def fileInfo(pathname="dataset/poloneix_data\\BTC_BTCD.csv-2014-07-01 00_00_00-2016-05-07 00_00_00"):
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
        # print("**************** Saving ", filename, "*************************")
        train_df = dataset[file]
        train_df.to_csv(
            path_or_buf=os.path.join(basepath, filename),
            sep=",", header=True)
class DataPreprocess:

    def __init__(self,
                 input_folder_name='../../dataset/updated_poloniex_data',
                 output_folder_name = '../../dataset/Poloneix_Preprocessed',
                 train_dates=[['2014-07-01', '2016-05-07'], ['2014-11-01', '2016-09-07'], ['2015-02-01', '2016-12-08'],
                              ['2015-05-01', '2017-03-07']],
                 test_dates = [['2016-05-07', '2016-06-27'], ['2016-09-07', '2016-10-28'], ['2016-12-08', '2017-01-28'],
                             ['2017-03-07', '2017-04-27']]):
        self.input_folder_name = input_folder_name
        self.output_folder_name = output_folder_name
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.assets = []
        self.dataset = {}

        self.dataset = self.load_dataset()
#        self.datasetContains = "" #Variable holding type of dataset in self.dataset "" means normal preprocessed, bffill_ means imputated processed

    def load_dataset(self, file_prefix="", dataset_path=""):
        # print("*******************Trying to load old files:")
        if dataset_path.strip() == "":
            dataset_path = self.output_folder_name
        file_name_list = []
        dataset = {}
        for start,end in self.train_dates:
            for path in  glob(os.path.join(dataset_path, file_prefix+"train_"+start+"*"+end+"*")):
                name = os.path.basename(path)
                dataset['train_'+start+'_'+end] = pd.read_csv(path, header=0, index_col=0)
                file_name_list += [name]
        for start,end in self.test_dates:
            for path in  glob(os.path.join(dataset_path, file_prefix+"test_"+start+"*"+end+"*")):
                name = os.path.basename(path)
                dataset['test_'+start+'_'+end] = pd.read_csv(path, header=0, index_col=0)
                file_name_list += [name]
        # if len(file_name_list) != 0:
            # print("******************* Following files were found to be present and loaded in dataset:" + "\n\t".join(file_name_list))
        return dataset


    def rl_load_train_test(self, feature_type=['open', 'low', 'high', 'close'],
                            datatype="bffill_", asset_name="BTC_XEM",
                           path="../../dataset/Poloneix_Preprocessednew", idx = 0,
                            ):
        rl_env = True
        history_train = []
        history_test = []
        if type(feature_type) == str:
            feature_type = [feature_type] # if there is only 1 feature jst convert it into 1 element list
        for feature in feature_type:
            h_train, h_test = self.load_train_test(asset_name=asset_name,
                                      feature_type=feature,
                                      rl_env=rl_env, path=path)
            history_train += [h_train]
            history_test += [h_test]
        history_train = np.concatenate(history_train, axis=-1)
        history_test = np.concatenate(history_test, axis=-1)
        return history_train, history_test


    def load_train_test(self, feature_type='open', datatype="bffill_",
                        asset_name="BTC_XEM",
                        path="../../dataset/Poloneix_Preprocessednew", idx = 0,
                        rl_env =False):
        dataset = self.load_dataset(file_prefix=datatype, dataset_path=path)
        train_data, test_data = {}, {}
        for key in dataset:
            s, start_date, end_date = key.split('_')
            cur_date = start_date+'_'+end_date

            if type(asset_name) == str:
                values = dataset[key][asset_name + '_' + feature_type]
            else:
                values = dataset[key][[asset + '_' + feature_type for asset in asset_name]]

            if s == "train":
                train_data[cur_date] = values
            else:
                test_data[cur_date] = values
        # for i in range(len(self.train_dates)):
        train_date = "_".join(self.train_dates[idx])
        test_date  = "_".join(self.test_dates[idx])
        # print("***************** Loading ", feature_type, " for Asset:", asset_name, " for data:", datatype, " for Train Dates:", train_date, " for Test Dates:", test_date)

        if rl_env == True:
            return helper.convert_to_env_data(train_data[train_date]), helper.convert_to_env_data(test_data[test_date])

        return train_data[train_date], test_data[test_date]

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
        # print("********************Preprocessing: ", file)
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
                else : pass
                    # print("************Warning: Data Missing for ", asset, " between ", start_date, " to ", end_date, "**************" )
            # print("********************************Saving File :"" between ", start_date, " to ", end_date, ' ***************************' )
            self.dataset[file+'_'+start_date+'_'+end_date] = train_df

if __name__=='__main__':
    input_folder_name = '../../dataset/poloneix_data'
    output_folder_name = '../../dataset/Poloneix_Preprocessednew'

    data = DataPreprocess(input_folder_name=input_folder_name,
                          output_folder_name=output_folder_name,
                          )
    # data.preprocess()
    # data.asset_names()
    # tr, te = data.load_train_test()
    # print(tr.head(2), te.head(2))
    tr, te = data.load_train_test(asset_name=['BTC_XEM','BTC_ETH'] ,feature_type='high',
                                path='../../dataset/Poloneix_Preprocessednew', rl_env=True)
    print(tr.shape, te.shape)