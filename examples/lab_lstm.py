from data_load.batchifier import Batchifier
from models.lstm_model import LSTMModel, tf
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=20
asset_list=ASSET_LIST
randomize_train=True
overlapping_train=False
IDX=0
NUM_EPOCHS = 10
VALIDATION_INTERVAL = 10
INIT_PV=1000
NUM_HID=20

if __name__ == '__main__':
    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSET_LIST, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)
    model = LSTMModel(num_hid=NUM_HID,bptt=BPTT)
    step = 1
    losses = []
    for epoch in range(1,NUM_EPOCHS + 1):
        for bTrainX, bTrainY in batch_gen.load_train():
            model.optimize(bTrainX, bTrainY)
            losses.append(model.loss(bTrainX, bTrainY).numpy()[0])
            step += 1
            if step % VALIDATION_INTERVAL == 0:
                allocation_wts = []
                price_change_vec = []
                for bEvalX, bEvalY in batch_gen.load_test():
                    pred_allocations = model.predict_portfolio_allocation(bEvalX)
                    assert bEvalY.shape == pred_allocations.shape
                    price_change_vec.append(bEvalY)
                    allocation_wts.append(pred_allocations)
                true_change_vec = np.concatenate(price_change_vec)
                allocation_wts = np.concatenate(allocation_wts)
                test_date = "_".join(batch_gen.dp.test_dates[IDX])
                m = get_metrics(dt_range=test_date)
                m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
