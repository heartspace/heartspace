import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.layers import xavier_initializer, batch_norm


def attention(query, key, value):
    # "Args:query: [batch_size, n, d_k], key.shape = [batch_size, m, d_k], value.shape = [batch_size, m, d_v]"
    # "Output:output.shape = [batch_size, n, d_k]"
    "Compute 'Scaled Dot Product Attention'"
    batch_size, n_query, d_k = query.get_shape().as_list()
    scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))/math.sqrt(d_k)#n*m
    p_attn = tf.nn.softmax(scores, axis = -1)
    return tf.matmul(p_attn, value), p_attn

class MultiHeadedAttention():
    # "Args:query: [batch_size, n, d_k], key.shape = [batch_size, m, d_k], value.shape = [batch_size, m, d_v]"
    # "Output:output.shape = [batch_size, n, d_k]"

    def __init__(self, h, d_model, dropout=0.1):
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def model(self, query, key, value, mask=None):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        xs = []
        batch_size, n_query, _ = tf.unstack(tf.shape(query))
        batch_size, n_key, _ = tf.unstack(tf.shape(key))
        batch_size, n_value, _ = tf.unstack(tf.shape(value))
        proj_query = tf.reshape(tf.layers.dense(query, self.d_model,
                                              kernel_regularizer=self.regularizer,name='output_query'),
                              [-1, n_query, self.h, self.d_k])
        proj_query = tf.reshape(tf.transpose(proj_query, [0, 2, 1, 3]), [-1, n_query, self.d_k])

        proj_key = tf.reshape(tf.layers.dense(key, self.d_model,
                                              kernel_regularizer=self.regularizer,name='output_key'),
                              [-1, n_key, self.h, self.d_k])
        proj_key = tf.reshape(tf.transpose(proj_key, [0, 2, 1, 3]), [-1, n_key, self.d_k])

        proj_value = tf.reshape(tf.layers.dense(value, self.d_model,
                                              kernel_regularizer=self.regularizer,name='output_value'),
                              [-1, n_value, self.h, self.d_k])
        proj_value = tf.reshape(tf.transpose(proj_value, [0, 2, 1, 3]), [-1, n_value, self.d_k])

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(proj_query, proj_key, proj_value)#[batch_size*n_head, n_query, d_v]

        # 3) "Concat" using a view and apply a final linear.
        x = tf.reshape(x, [-1, self.h, n_query, self.d_k])
        x = tf.transpose(x, [0, 2, 1, 3])#[batch_size, n_query, n_head, d_v]
        x = tf.reshape(x, [-1, n_query, self.h*self.d_k])
        return tf.layers.dense(x, self.d_model, activation=None,trainable=True,
                               kernel_regularizer=self.regularizer,name='mult_head')

    def PositionwiseFeedForward(self, x, d_model, d_diff):
        l1 = tf.layers.dense(x, d_diff, activation = tf.nn.relu, kernel_regularizer=self.regularizer,name='feed1')
        # l1 = tf.dropout(l1, keep_rate = 0.8)
        l2 = tf.layers.dense(l1, d_model, kernel_regularizer=self.regularizer,name='feed2')
        return l2

    def SublayerConnection(self, x, keep_rate):
        return batch_norm((x + tf.dropout(self.model(x, x, x), keep_rate = keep_rate)))


if __name__ == '__main__':
    vocab_size = 10
    embedding_dim = 20
    batch_num = 2
    query = tf.Variable(tf.constant(0.0, shape=[batch_num, vocab_size, embedding_dim]),
                    trainable=False, name="query").assign(np.random.randint(0, high=20, size=[batch_num, vocab_size, embedding_dim], dtype='l'))
    key = tf.Variable(tf.constant(0.0, shape=[batch_num, vocab_size, embedding_dim]),
                    trainable=False, name="key").assign(np.random.randint(0, high=20, size=[batch_num, vocab_size, embedding_dim], dtype='l'))
    value = tf.Variable(tf.constant(0.0, shape=[batch_num, vocab_size, embedding_dim]),
                    trainable=False, name="value").assign(np.random.randint(0, high=20, size=[batch_num, vocab_size, embedding_dim], dtype='l'))

    mult_head = MultiHeadedAttention(h = 4, d_model = embedding_dim)
    output = mult_head.model(query, key, value)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output_ = sess.run(output)
        print(output_)