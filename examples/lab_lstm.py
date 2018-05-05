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
BPTT=10
asset_list=ASSET_LIST
randomize_train=True # Always set to false if replay != 0
replay=0
overlapping_train=True
IDX=0
NUM_EPOCHS = 100
INIT_PV=1000
NUM_HID=200
ASSETS = ASSET_LIST
LR = 1e-4
MODEL = "CNN" # Set "LSTM" or "CNN"


if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)

    model = LSTMModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR, bsz=BSZ, clip_norm=5.0)
    if MODEL == "CNN":
        model = CNNModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR, bsz=BSZ, clip_norm=5.0)

    buffer = ReplayBuffer(buffer_size=replay)
    with tf.Session() as sess:
        sess.run(model.tf_init())
        losses = []
        for epoch in range(1,NUM_EPOCHS + 1):
            for bTrainX, bTrainY in batch_gen.load_train():
                if replay == 0:
                    _, loss = sess.run([model.optimize, model.loss], feed_dict={
                        model.data: bTrainX, model.target: bTrainY
                    })
                else:
                    if buffer.size < buffer.max_size:
                        _, loss = sess.run([model.optimize, model.loss], feed_dict={
                            model.data: bTrainX, model.target: bTrainY
                        })
                        buffer.add([bTrainX, bTrainY], bsz=len(bTrainX))
                    else:
                        vars = buffer.get_batch(bsz=BSZ)
                        _, loss = sess.run([model.optimize, model.loss], feed_dict={
                            model.data: vars[0], model.target: vars[1]
                        })

                losses.append(loss)
            print("Epoch {} Average Train Loss: {}, validating...".format(epoch, sum(losses)/len(losses)))
            losses = []
            allocation_wts = []
            price_change_vec = []
            for bEvalX, bEvalY in batch_gen.load_test():
                # if replay == 0:
                pred_allocations = sess.run(model.predict_portfolio_allocation,
                                            feed_dict={
                                                model.data: bEvalX
                                            })
                assert bEvalY.shape == pred_allocations.shape
                price_change_vec.append(bEvalY)
                allocation_wts.append(pred_allocations)

                # else:
                #     if buffer.size < buffer.max_size:
                #         pred_allocations = sess.run(model.predict_portfolio_allocation,
                #                                     feed_dict={
                #                                         model.data: bEvalX
                #                                     })
                #         assert bEvalY.shape == pred_allocations.shape
                #         price_change_vec.append(bEvalY)
                #         allocation_wts.append(pred_allocations)
                #         buffer.add([bEvalX, bEvalY], bsz=len(bEvalX))
                #     else:
                #         vars = buffer.get_batch(bsz=BSZ)
                #         pred_allocations = sess.run(model.predict_portfolio_allocation,
                #                                     feed_dict={
                #                                         model.data: vars[0]
                #                                     })
                #         assert bEvalY.shape == pred_allocations.shape
                #         price_change_vec.append(vars[1])
                #         allocation_wts.append(pred_allocations)

            true_change_vec = np.concatenate(price_change_vec)
            allocation_wts = np.concatenate(allocation_wts)

            random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
            test_date = "_".join(batch_gen.dp.test_dates[IDX])
            m = get_metrics(dt_range=test_date)
            print("Our Policy:")
            m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
            # print("Random Policy:")
            # m.apv_multiple_asset(true_change_vec, random_alloc_wts, get_graph=False, pv_0=INIT_PV)
