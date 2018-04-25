from data_load.batchifier import Batchifier
from models.lstm_model import LSTMModel, tf
from literals import ASSET_LIST

DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=16
BPTT=50
asset_list=ASSET_LIST
randomize_train=False
IDX=0

if __name__ == '__main__':
    log_at = 50
    batch_gen = Batchifier(data_path=DATA_PATH, bsz=BSZ, bptt=BPTT, idx=IDX,
                           asset_list=ASSET_LIST, randomize_train=randomize_train)
    model = LSTMModel(num_hid=20,bptt=200)
    with tf.Session() as sess:
        sess.run(model.tf_init())
        for epoch in range(1,10):
            train_batch_gen = batch_gen.load_train()
            final, apv = [], []
            for bTrainX, bTrainY in train_batch_gen:
                sess.run(model.optimize, feed_dict = {
                    model.data: bTrainX,
                    model.target: bTrainY,
                    model._is_training: True
                })
                apv.append(sess.run(model.apv, {
                    model.data: bTrainX,
                    model.target: bTrainY,
                    model._is_training: False
                }))
                if (len(apv) + 1) % log_at == 0:
                    print("Minibatch({}) APV at epoch {} = {}".format(log_at * (len(final) + 1), epoch, sum(apv)/len(apv)))
                    final.append(sum(apv)/len(apv))
                    apv = []
            final.append(sum(apv) / len(apv))
            print("APV at epoch {} = {}".format(epoch, sum(final)/len(final)))
