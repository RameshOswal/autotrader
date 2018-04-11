from data_load.batchifier import batchify
from models.lstm_model import LSTMModel, tf

if __name__ == '__main__':
    log_at = 50
    batch_gen = batchify("../dataset/Poloneix_Preprocessednew")
    model = LSTMModel(num_hid=20,bptt=200)
    with tf.Session() as sess:
        sess.run(model.tf_init())
        for epoch in range(1,10):
            train_batch_gen = batch_gen.load_train(bptt=200, normalize=False, bsz=128, idx = 0)
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
