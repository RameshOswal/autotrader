import os

from glob import glob
import pandas as pd
import os
import  numpy as np
from data_load import helper

class DataPreprocess:
    def __init__(self, input_folder_name='../../dataset/stock_data',
                 output_folder_name='../../dataset/stock_data_Preprocessed',
                 processed_file_name = 'preprocessed_Stock_data.csv'):
        self.input_folder_name = input_folder_name
        self.output_folder_name = output_folder_name
        self.train_dates = ['stock_date1']
        self.test_dates = ['stock_date1']
        self.processed_file_name  = processed_file_name
        self.dataset = pd.DataFrame()


    def asset_names(self):
        all_columns = self.dataset.columns
        assets = set()
        for col in all_columns:
            cols = col.split('_')
            if len(cols) > 1: #to avoid date columns name
                assets.add(cols[0])
        self.assets = assets
        return self.assets


    def save_dataset(self, filename):
        filename = os.path.join(self.output_folder_name,filename)
        self.dataset.to_csv(filename)


    def preprocess(self):
        dataset = pd.DataFrame()
        for path in glob(self.input_folder_name+'/*'):
            filename = os.path.basename(path)
            stock_name = filename.split('_')[0] # e.g. filename is AAPL_stock_price
            df = pd.read_csv(path, header=0)
            df.columns = [str.upper(stock_name+'_'+col_name) if col_name!='Date' else str.upper(col_name) for col_name in df.columns]
            dataset = dataset.merge(df, how='outer', on='DATE') if dataset.size != 0 else df
        self.dataset = dataset
        self.save_dataset(self.processed_file_name)


    def load_preprocessed(self, path = ""):
        if path.strip() == "":
            path = os.path.join(self.output_folder_name, self.processed_file_name)
        self.dataset = pd.read_csv(path)

        
    def rl_load_train_test(self, feature_type=['OPEN'], asset_name=["FB"], path="", train_test_ratio=0.8):
        rl_env = True
        history_train = []
        history_test = []
        if type(feature_type) == str:
            feature_type = [feature_type] # if there is only 1 feature jst convert it into 1 element list
        for feature in feature_type:
            h_train, h_test = self.load_train_test(asset_name=asset_name,
                                      feature_type=feature, train_test_ratio=train_test_ratio,
                                      rl_env=rl_env, path=path)
            history_train += [h_train]
            history_test += [h_test]
        history_train = np.concatenate(history_train, axis=-1)
        history_test = np.concatenate(history_test, axis=-1)
        return history_train, history_test
    
    def load_train_test(self, feature_type='OPEN', asset_name="FB", path="", train_test_ratio=0.8, rl_env=False):
        """

        :param feature_type: Name of Feature like open, close etc.
        :param asset_name: names of stock asset we want
        :param path: path of the preprocessed file
        :param train_test_ratio:
        :param idx:
        :return:
        """
        if path.strip() == "":
            path = os.path.join(self.output_folder_name, self.processed_file_name)

        if self.dataset.size == 0:
            self.load_preprocessed(path)

        if type(asset_name) == str:
            value = self.dataset[asset_name+"_"+feature_type]
        else: #For multuple assets_data
            value = self.dataset[[name+"_"+feature_type for name in asset_name]]

        row_count = value.shape[0]
        num_train_rows = int(row_count * train_test_ratio)
        if rl_env == True:
            value = helper.convert_to_env_data(value)
            return value[:,0:num_train_rows,:], value[:,num_train_rows:,:]
        return value.iloc[0:num_train_rows], value.iloc[num_train_rows:]

if __name__=='__main__':
    input_folder_name = '../../dataset/stock_data'
    output_folder_name = '../../dataset/stock_data_Preprocessed'

    data = DataPreprocess(input_folder_name=input_folder_name,
                          output_folder_name=output_folder_name,
                          )
    # data.preprocess()
    # data.asset_names()
    # tr, te = data.load_train_test()
    # print(tr.head(2), te.head(2))
    tr, te = data.load_train_test(asset_name=['FB','AAPL'] ,feature_type='OPEN',
                                  train_test_ratio=0.8, path='../../dataset/stock_data_Preprocessed/preprocessed_Stock_data.csv', rl_env=True)
    print(tr.shape, te.shape)