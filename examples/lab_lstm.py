import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_load.load_stocks as stocks
from data_load.batchifier import Batchifier
from data_load.replay_buffer import ReplayBuffer
from models.lstm_model import LSTMModel, tf
from models.lstm_cnn_model import LSTMCNNModel
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
from sklearn.utils.extmath import softmax




DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=10
asset_list=ASSET_LIST
randomize_train=True # Always set to false if replay = True
replay=128
overlapping_train=True
IDX=0
NUM_EPOCHS = 100
INIT_PV=1000
NUM_HID=20
ASSETS = ASSET_LIST
LR = 1e-4

# DATA_PATH = "../../dataset/stock_data_Preprocessed"
# BSZ=32
# BPTT=10
# dp = stocks.DataPreprocess()
#    # dp.load_preprocessed('../../dataset/stock_data_Preprocessed/preprocessed_Stock_data.csv')
# dp.load_preprocessed(DATA_PATH+'/preprocessed_Stock_data.csv')
# stocks_name = dp.asset_names()
# print(stocks_name)
# asset_list=stocks_name

if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)
    # batch_gen = Batchifier(data_path='../../dataset/5yrs_preprocessed/all_5_yrs_processed.csv',
    #                   asset_list=stocks_name, data_preprocess=dp,
    #                   idx=0,
    #                    randomize_train=True,
    #                    overlapping_train=True, bptt = BPTT
    #                   )
    model = LSTMCNNModel(num_hid=NUM_HID,bptt=BPTT, num_assets=len(asset_list), lr=LR, bsz=BSZ, clip_norm=5.0)
    buffer = ReplayBuffer(buffer_size=replay)
    ops = model.logits()

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook("./log_dir", save_steps=1000, saver=saver)
    hooks = [checkpoint_saver_hook]
    sess = tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir="./log_dir")

    losses = []
    train_ops = [
        ops["gs"],
        ops["train_op"],
        ops["loss"]
    ]

    test_ops = [
        ops["alloc_weights"]
    ]

    for epoch in range(1,NUM_EPOCHS + 1):
        for bTrainX, bTrainY in batch_gen.load_train():
            g, _, loss = sess.run(train_ops, feed_dict = {ops["net"]: bTrainX, ops["target"]: bTrainY})
            losses.append(loss)

        print("Epoch {} Average Train Loss: {}, validating...".format(epoch, sum(losses)/len(losses)))
        losses = []
        allocation_wts = []
        price_change_vec = []
        for bEvalX, bEvalY in batch_gen.load_test():
            pred_allocations = sess.run(test_ops, feed_dict = {ops["net"]: bEvalX})
            assert bEvalY.shape == pred_allocations[0].shape
            price_change_vec.append(bEvalY)
            allocation_wts.append(pred_allocations[0])
        true_change_vec = np.concatenate(price_change_vec)
        allocation_wts = np.concatenate(allocation_wts)

        # print("Overall change in price:", list(np.prod(true_change_vec, axis=0)))
        # print("Mean Asset Allocation:", list(np.mean(allocation_wts, axis=0)))
        random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
        test_date = "_".join(batch_gen.dp.test_dates[IDX])
        m = get_metrics(dt_range=test_date)
        print("Our Policy:")
        m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
        print("Random Policy:")
        m.apv_multiple_asset(true_change_vec, random_alloc_wts, get_graph=True, pv_0=INIT_PV)