import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_load.load_stocks as stocks
from data_load.batchifier import Batchifier
from models.cnn_model import CNNModel, tf
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
from sklearn.utils.extmath import softmax




DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=10
asset_list=ASSET_LIST
randomize_train=True
overlapping_train=True
IDX=0
NUM_EPOCHS = 100
INIT_PV=1000
NUM_HID=20
ASSETS = ASSET_LIST
LR = 1e-4


if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)

    model = CNNModel(num_hid=NUM_HID, bptt=BPTT, num_assets=len(asset_list), lr=LR, clip_norm=5.0)

    with tf.Session() as sess:
        sess.run(model.tf_init())
        losses = []
        for epoch in range(1,NUM_EPOCHS + 1):
            for bTrainX, bTrainY in batch_gen.load_train():
                loss = model.optimize(sess, bTrainX, bTrainY)
                losses.append(loss)
            print("Epoch {} Average Train Loss: {}, validating...".format(epoch, sum(losses)/len(losses)))
            losses = []
            allocation_wts = []
            price_change_vec = []
            for bEvalX, bEvalYFat in batch_gen.load_test():
                bEvalY = bEvalYFat[:,-1,:]
                pred_allocations = model.predict_allocation(sess, bEvalX)
                assert bEvalY.shape == pred_allocations.shape
                price_change_vec.append(bEvalY)
                allocation_wts.append(pred_allocations)

            true_change_vec = np.concatenate(price_change_vec)
            allocation_wts = np.concatenate(allocation_wts)

            random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
            test_date = "_".join(batch_gen.dp.test_dates[IDX])
            m = get_metrics(dt_range=test_date)
            print("Our Policy:")
            m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)