from data_load.batchifier import batchify
from models.lstm_model import LSTMModel, tf

if __name__ == '__main__':
    batch_gen = batchify()
    model = LSTMModel(num_hid=9)
    train_batch_gen = batch_gen.load_train()
    with tf.Session() as sess:
        model.tf_init()
        for bTrainX, bTrainY in train_batch_gen:
            sess.run(model.optimize, feed_dict = {
                model.data: bTrainX,
                model.target: bTrainY
            })
            print(-sess.run(model.loss, {
                model.data: bTrainX,
                model.target: bTrainY
            }))
