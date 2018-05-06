import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_load.load_stocks as stocks
from data_load.batchifier import Batchifier
from data_load.replay_buffer import ReplayBuffer
from models.lstm_model import LSTMModel, tf
from models.cnn_model import CNNModel
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
from sklearn.utils.extmath import softmax




DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
RB_FACTOR = 3 # REPLAY BUFFER FACTOR
BPTT=10
asset_list=ASSET_LIST

MODEL = "CNN" # Set "LSTM" or "CNN"
NUM_HID=20
IDX=0 # Date Index
TEST_OPTIMIZE_AFTER=10
LOG_AFTER=5000

randomize_train=False # Always set to false if replay != 0
overlapping_train=True

replay=BSZ*RB_FACTOR
NUM_EPOCHS = 100
INIT_PV=1000
ASSETS = ASSET_LIST
LR = 1e-4


if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=1, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)

    if MODEL == "LSTM":
        model = LSTMModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR,
                            scope_prefix="model", clip_norm=5.0)
        testModel = LSTMModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR,
                            scope_prefix="testModel", clip_norm=5.0)
    elif MODEL == "CNN":
        model = CNNModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR,
                          scope_prefix="model", clip_norm=5.0)
        testModel = CNNModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR,
                          scope_prefix="testModel", clip_norm=5.0)

    buffer = ReplayBuffer(buffer_size=replay)

    with tf.Session() as sess:
        sess.run(model.tf_init())
        losses = []
        for epoch in range(1,NUM_EPOCHS + 1):

            batch_losses = 0.0
            for bTrainX, bTrainY in batch_gen.load_train():
                if buffer.size < buffer.max_size:
                    buffer.add(state=bTrainX, action=bTrainY)
                    continue
                else:
                    buffer.add(state=bTrainX, action=bTrainY)
                    state, reward, action = buffer.get_batch(bsz=BSZ)
                    _, loss = sess.run([model.optimize, model.loss], feed_dict={
                        model.data: state, model.target: action
                    })
                    losses.append(loss)

                if len(losses) % LOG_AFTER == 0:
                    print("Loss after Mini Batch {} Epoch {} = {}".format(len(losses)/LOG_AFTER, epoch, batch_losses / LOG_AFTER))
                    batch_losses = 0.0
                else:
                    batch_losses += loss

            print("Epoch {} Average Train Loss: {}, validating...".format(epoch, sum(losses)/len(losses)))
            losses = []
            allocation_wts = []
            price_change_vec = []

            for var_idx in range(len(testModel.train_vars)//2): testModel.train_vars[var_idx + len(testModel.train_vars)//2].assign(testModel.train_vars[var_idx])

            testLoss = []
            for idx, (bEvalX, bEvalY) in enumerate(batch_gen.load_test()):
                if buffer.size < buffer.max_size:
                    buffer.add(state=bEvalX, action=bEvalY)
                    continue
                else:
                    buffer.add(state=bEvalX, action=bEvalY)
                    pred_allocations = sess.run(model.predict_portfolio_allocation,
                                                feed_dict={
                                                    model.data: bEvalX
                                                })
                    assert bEvalY[:, -1, :].shape == pred_allocations.shape, "{} and {}".format(bEvalY.shape, pred_allocations.shape)

                    price_change_vec.append(bEvalY[:, -1, :])
                    allocation_wts.append(pred_allocations)

                    if (idx + 1) % TEST_OPTIMIZE_AFTER == 0:
                        state, reward, action = buffer.get_batch(bsz=BSZ)
                        _, tloss = sess.run([testModel.optimize, testModel.loss],
                                                    feed_dict={
                                                        testModel.data: state,
                                                        testModel.target: action
                                                    })
                        testLoss.append(tloss)
            print("Epoch {} Average Test Loss: {}, ".format(epoch, sum(testLoss)/len(testLoss)))

            true_change_vec = np.concatenate(price_change_vec)
            allocation_wts = np.concatenate(allocation_wts)

            # random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
            test_date = "_".join(batch_gen.dp.test_dates[IDX])
            m = get_metrics(dt_range=test_date)
            print("Our Policy:")
            m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
            # print("Random Policy:")
            # m.apv_multiple_asset(true_change_vec, random_alloc_wts, get_graph=False, pv_0=INIT_PV)

            buffer.clear()