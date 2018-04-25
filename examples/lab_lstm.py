from data_load.batchifier import Batchifier
from models.lstm_model import LSTMModel, tf
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics

DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=128
BPTT=50
asset_list=ASSET_LIST
randomize_train=False
IDX=0
NUM_EPOCHS = 10
VALIDATION_INTERVAL = 1
INIT_PV=1000

if __name__ == '__main__':
    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSET_LIST, randomize_train=randomize_train)
    model = LSTMModel(num_hid=20,bptt=BPTT)
    with tf.Session() as sess:
        sess.run(model.tf_init())
        train_avgapv = []
        step = 1
        for epoch in range(1,NUM_EPOCHS + 1):
            for bTrainX, bTrainY in batch_gen.load_train():
                sess.run(model.optimize, feed_dict = {
                    model.data: bTrainX,
                    model.target: bTrainY,
                    model._is_training: True
                })
                step += 1
                if step % VALIDATION_INTERVAL == 0:
                    allocation_wts = []
                    price_change_vec = []
                    for bEvalX, bEvalY in batch_gen.load_test():
                        pred_allocations = sess.run(model.predict_portfolio_allocation, {
                            model.data: bEvalX,
                            model._is_training: False
                        })
                        assert bEvalY.shape == pred_allocations.shape
                        price_change_vec.append(bEvalY)
                        allocation_wts.append(pred_allocations)
                    true_change_vec = np.concatenate(price_change_vec)
                    allocation_wts = np.concatenate(allocation_wts)
                    test_date = "_".join(batch_gen.dp.test_dates[IDX])
                    m = get_metrics(dt_range=test_date)
                    m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
            #         print("Minibatch({}) APV at epoch {} = {}".format(LOG_INTERVAL * (len(final) + 1), epoch, sum(apv)/len(apv)))
            #         final.append(sum(apv)/len(apv))
            #         apv = []
            # final.append(sum(apv) / len(apv))
            # print("APV at epoch {} = {}".format(epoch, sum(final)/len(final)))
