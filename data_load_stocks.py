import Literals
import os

from glob import glob
import pandas as pd
import os
class DataPreprocess:
    def __init__(self, input_folder_name, output_folder_name, processed_file_name = 'preprocessed_Stock_data.csv'):
        self.input_folder_name = input_folder_name
        self.output_folder_name = output_folder_name
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
        print("***************Saving DataSet***********")
        filename = os.path.join(self.output_folder_name,filename)
        self.dataset.to_csv(filename)


    def preprocess(self):
        dataset = pd.DataFrame()
        for path in glob(self.input_folder_name+'/*'):
            filename = os.path.basename(path)
            stock_name = filename.split('_')[0] # e.g. filename is AAPL_stock_price
            print("******************Loading Stock:" + stock_name+" *****************")
            df = pd.read_csv(path, header=0)
            print("*******************File Size: ", df.shape)
            df.columns = [str.upper(stock_name+'_'+col_name) if col_name!='Date' else str.upper(col_name) for col_name in df.columns]
            dataset = dataset.merge(df, how='outer', on='DATE') if dataset.size != 0 else df
        self.dataset = dataset
        self.save_dataset(self.processed_file_name)


    def load_preprocessed(self, path = ""):
        if path.strip() == "":
            path = os.path.join(self.output_folder_name, self.processed_file_name)
        self.dataset = pd.read_csv(path)
    def load_train_test(self, feature_type='OPEN', asset_name="FB",path="", train_test_ratio=0.8):
        if path.strip() == "":
            path = os.path.join(self.output_folder_name, self.processed_file_name)

        if self.dataset.size == 0:
            self.load_preprocessed(path)

        if type(feature_type) == str:
            value = self.dataset[asset_name+"_"+feature_type]
        else: #For multuple assets_data
            value = self.dataset[[asset_name+"_"+feature_name for feature_name in feature_type]]

        row_count = value.shape[0]
        num_train_rows = int(row_count * train_test_ratio)
        return value.iloc[0:num_train_rows], value.iloc[num_train_rows:]

if __name__=='__main__':

    input_folder_name = 'dataset/stock_data'
    output_folder_name = 'dataset/stock_data_Preprocessed'

    data = DataPreprocess(input_folder_name=input_folder_name,
                          output_folder_name=output_folder_name,
                          )
    # data.preprocess()
    data.asset_names()
    tr, te = data.load_train_test()
    print(tr.head(2), te.head(2))
    tr, te = data.load_train_test(feature_type=['OPEN','CLOSE'], train_test_ratio=0.5, path='preprocessed_Stock_data.csv')
    print(tr.head(2), te.head(2))