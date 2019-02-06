import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.layers import xavier_initializer, batch_norm
from tensorflow.contrib import rnn


class seq2seqNN(object):
    def __init__(self, **kwargs):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)



    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob):

        # stacked_cells = tf.contrib.rnn.MultiRNNCell(
        #     [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
        stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(rnn_size), keep_prob) for _ in range(num_layers)])


        outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                           rnn_inputs,
                                           dtype=tf.float32)
        return outputs, state

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             output_layer = 1, keep_prob = 1.0):

        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                                 output_keep_prob=keep_prob)

        # for only input layer
        input_n, seq_len, input_dim = tf.unstack(tf.shape(dec_embed_input))
        target_sequence_length = tf.tile([seq_len],[input_n],name=None)
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                                   target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                  helper,
                                                  encoder_state,
                                                  output_layer)

        # unrolling the decoder layer
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=seq_len)
        return outputs

    def decoding_layer(self, dec_input, encoder_state,rnn_size,
                       num_layers, layer):

        # cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)])

        with tf.variable_scope("decode"):
            train_output, _ = self.decoding_layer_train(encoder_state, cells, dec_input,output_layer = layer, keep_prob = 1.0)
        return train_output


    def exp_seq2seq(self, input, keep_prob, rnn_size, output_dim, num_layers):
        output = input
        enc_outputs, enc_states = self.encoding_layer(input, rnn_size, num_layers, keep_prob)
        output_layer = tf.layers.Dense(output_dim, activation=None)
        train_output = self.decoding_layer(output, enc_states, rnn_size, num_layers, output_layer)
        loss = tf.reduce_mean(tf.squared_difference(input, train_output))
        opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        bottleneck = tf.reduce_mean(enc_states, axis = 0)
        return train_output, tf.shape(enc_states), loss, opt

    def seq2seq(self, input, keep_prob, rnn_size, output_dim, num_layers):
        output = input
        enc_outputs, enc_states = self.encoding_layer(input, rnn_size, num_layers, keep_prob)
        output_layer = tf.layers.Dense(output_dim, activation=None)
        train_output = self.decoding_layer(output, enc_states, rnn_size, num_layers, output_layer)
        bottleneck = tf.reduce_mean(enc_states, axis=0)
        return train_output, bottleneck




if __name__ == '__main__':
    vocab_size = 10
    embedding_dim = 64
    batch_num = 3
    # query = tf.Variable(tf.constant(0.0, shape=[batch_num, vocab_size, embedding_dim]),
    #                 trainable=False, name="query").assign(np.random.randint(0, high=20, size=[batch_num, vocab_size, embedding_dim], dtype='l'))
    query = tf.Variable(tf.constant(0.0, shape=[batch_num, vocab_size, embedding_dim]),
                    trainable=False, name="query").assign(np.random.rand(batch_num, vocab_size, embedding_dim))
    seq2seq = seq2seqNN()
    output, state_shape, loss, opt = seq2seq.exp_seq2seq(query, keep_prob = 1.0,
                              rnn_size = 64, output_dim = embedding_dim, num_layers = 4)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(200):
            output_, state_shape_, loss_, _ = sess.run([output, state_shape, loss, opt])
            if i%100==0:
                print(np.round(output_,2), state_shape_, loss_)