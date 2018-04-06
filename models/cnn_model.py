__author__ = "deeptrader"

import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import numpy as np
from autotrader.data_load.batchifier import batchify


class CNN:
    def __init__(self):
        self.conv_dims = [256, 256, 256, 256, 256, 256]
        self.kernel_size = [4, 4, 4, 4, 4, 4]
        self.strides = [1, 1, 1, 1, 1, 1]
        self.full = [1024, 512]
        self.num_outputs = 2
        self.learning_rate = 0.001
        self.num_epochs = 10

    def build_graph(self):
        X = tf.placeholder(dtype = tf.int32, shape = [None, 309, 276, 1])
        y = tf.placeholder(dtype = tf.int32, shape = [None])

        with tf.variable_scope("CNN", reuse = tf.AUTO_REUSE):
            image_x = tf.to_float(X)
            for idx, layer_size in enumerate(self.conv_dims):
                image_x = tfcl.conv2d(inputs = image_x, num_outputs = layer_size, kernel_size = [self.kernel_size[idx], self.kernel_size[idx]],
                                      stride = [self.strides[idx], self.strides[idx]], padding = "SAME")

        with tf.variable_scope("LINEAR", reuse = tf.AUTO_REUSE):
            x = tf.reshape(image_x, [-1, 309 * 276 * self.conv_dims[-1]])
            for idx, layer_size in enumerate(self.full):
                x = tfcl.fully_connected(inputs = x, num_outputs = layer_size, weights_initializer=tfcl.xavier_initializer(), activation_fn = tf.nn.relu)

            logits = tfcl.fully_connected(inputs = x, num_outputs = self.num_outputs, activation_fn = None)


        loss_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        start_training = optimizer.minimize(loss_softmax)

        return {
            "X" : X,
            "y" : y,
            "start_training" : start_training,
            "loss" : loss_softmax
        }

    def main(self):
        graph = tf.Graph()
        with graph.as_default():
            ops = self.build_graph()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.num_epochs):
                    print("Training at epoch {}".format(epoch + 1))
                    for b_X, b_y in dl.get_data():
                        waste, loss = sess.run([ops["start_training"], ops["loss"]], feed_dict = {
                            ops["X"] : b_X, ops["y"] : b_y
                        })
                        print(loss)



if __name__ == '__main__':
    dl = Dataloader()

    # for file in dl.get_data():
    #     image = Image.open("Abnormal/resized/{}".format(file))
    #     print(image.size)
    bas = baseline()
    bas.main()



if __name__ == '__main__':
    pass


