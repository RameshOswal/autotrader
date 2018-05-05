import numpy as np
import pandas as pd

def convert_to_env_data(df):
    assets = list(set(["_".join(i.split('_')[:-1]) for i in df.columns]))
    ret = np.array([])
    for asset in assets:
        temp = df[ [ colname for colname in df.columns if asset in colname ]].values
        if ret.size == 0:
            ret = temp
        else:
            ret = np.concatenate((ret, temp), axis=0)
    return ret.reshape(len(assets), df.shape[0], -1)

def test_convert_to_env_data():
    df = pd.DataFrame({'a_1':[1,2,3,4,5],
                       'a_2':range(10,15),
                       'b_1': [5,4,3,2,1],
                       'b_2': range(10, 15)
                       })
    ret = convert_to_env_data(df)
    actual_ret = np.array([[[1,10], [2,11], [3,12], [4,13], [5,14]],
                           [[5, 10], [4, 11], [3, 12], [2, 13], [1, 14]]])
    assert np.all(ret==actual_ret)


if __name__=='__main__':
    test_convert_to_env_data()