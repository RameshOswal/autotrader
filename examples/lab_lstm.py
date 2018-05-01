from data_load.batchifier import Batchifier
from models.lstm_model import LSTMModel, tf
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
import tensorflow.contrib.eager as tfe
from sklearn.utils.extmath import softmax
tfe.enable_eager_execution()


DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=50
asset_list=ASSET_LIST
randomize_train=False
overlapping_train=True
IDX=0
NUM_EPOCHS = 10
INIT_PV=1000
NUM_HID=20
ASSETS = ASSET_LIST
LR = 1

if __name__ == '__main__':
    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)
    model = LSTMModel(num_hid=NUM_HID,bptt=BPTT, num_assets=len(ASSETS), lr=LR)
    losses = []
    for epoch in range(1,NUM_EPOCHS + 1):
        for bTrainX, bTrainY in batch_gen.load_train():
            model.optimize(bTrainX, bTrainY)
            losses.append(model.loss(bTrainX, bTrainY).numpy())
        print("Epoch {} Average Train Loss: {}, validating...".format(epoch, np.mean(losses)))
        losses = []
        allocation_wts = []
        price_change_vec = []
        for bEvalX, bEvalY in batch_gen.load_test():
            pred_allocations = model.predict_portfolio_allocation(bEvalX)
            assert bEvalY.shape == pred_allocations.shape
            price_change_vec.append(bEvalY)
            allocation_wts.append(pred_allocations)
        true_change_vec = np.concatenate(price_change_vec)
        allocation_wts = np.concatenate(allocation_wts)
        # print("Overall change in price:", list(np.prod(true_change_vec, axis=0)))
        # print("Mean Asset Allocation:", list(np.mean(allocation_wts, axis=0)))
        random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
        test_date = "_".join(batch_gen.dp.test_dates[IDX])
        m = get_metrics(dt_range=test_date)
        print("Our Policy:")
        m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
        # print("Random Policy:")
        # m.apv_multiple_asset(true_change_vec, random_alloc_wts, get_graph=True, pv_0=INIT_PV)
