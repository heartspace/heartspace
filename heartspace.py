import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
from tensorflow.contrib.layers import xavier_initializer, batch_norm
from kdd_model.componet.position_embedding import *
from kdd_model.componet.mult_head_attention import *


def convert_image_data_to_float(image_raw):
    img_float = tf.div(tf.cast(image_raw, tf.float32), 100.0)
    return img_float

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class NN(object):
    def __init__(self, **kwargs):
        self.num_input = kwargs['num_input']
        self.num_pos = kwargs['num_pos']
        self.num_neg = kwargs['num_neg']
        self.n_user = kwargs['n_user']
        self.height, self.width = kwargs['featureDimension']
        self.embed_size = kwargs['embed_size']
        self.pr = kwargs['penalty_rate']
        self.alpha = kwargs['alpha']
        self.lr = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']
        self.v_scope = kwargs['v_scope']
        self.attention_size = kwargs['attention_size']
        self.n_filters = kwargs['n_filters']#[2, 4, 8, 10, 12]

        self.filter_sizes = kwargs['filter_sizes']#[5, 5, 5, 5, 5]
        self.image_featuredim = kwargs['image_featuredim']#image_featuredim = 100
        self.n_mlp = kwargs['n_mlp']
        self.mask_filter = kwargs['mask_filter']
        self.overall_indicator = kwargs['overall_indicator']
        self.n_class = kwargs['n_class']
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def encode_block(self, conv_output, kernel, layer_id, n_output):
        conv_output1 = tf.layers.conv1d(conv_output,
                                       filters=n_output,
                                       kernel_size=kernel,
                                       strides=1,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.contrib.layers.xavier_initializer(),
                                       kernel_regularizer=self.regularizer,
                                       trainable=True,
                                       name='layer1_%s' % layer_id)

        conv_output = tf.layers.max_pooling1d(conv_output1,
                                              pool_size=3,
                                              strides=2,
                                              padding='same',
                                              name='layer_%s_maxpool' % layer_id)
        return conv_output


    def decode_block(self, input, kernel_size, depth_output, stride, activation, name):
        dim0, dim1, dim2 = input.get_shape().as_list()  # batch_size, cols, depth
        with tf.variable_scope(name):
            k = tf.get_variable('kernel', shape = [kernel_size, depth_output, dim2],
                                dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())  # note k.shape = [cols, depth_output, depth_in]
            b = tf.get_variable(name='b',
                                shape=depth_output,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            output_shape = tf.stack([tf.shape(input)[0], dim1*2, depth_output])
            output = tf.contrib.nn.conv1d_transpose(value=input,
                                          filter=k,
                                          output_shape=output_shape,
                                          stride=stride,
                                          padding='SAME')
            output = activation(tf.nn.bias_add(output, b))
        return output

    def squeeze_excite_block_channel(self, input, filters, name, ratio=1):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''

        init = input

        se = tf.reduce_mean(init, axis = 1)
        se = tf.layers.dense(se, filters // ratio, activation=tf.nn.relu,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se1'+name)
        se = tf.layers.dense(se, filters, activation=tf.nn.sigmoid,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='se2'+name)

        x = tf.multiply(init, tf.expand_dims(se, axis = 1))
        return x

    def squeeze_excite_block_temporal(self, input, filters, name, ratio=1):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''

        init = input

        se = tf.reduce_mean(init, axis = 2)
        se = tf.layers.dense(se, filters // ratio, activation=tf.nn.relu,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='tse1'+name)
        se = tf.layers.dense(se, filters, activation=tf.nn.sigmoid,
                              trainable=True,
                              kernel_regularizer=self.regularizer,
                              use_bias = False,
                              name='tse2'+name)

        x = tf.multiply(init, tf.expand_dims(se, axis = 2))
        return x

    def encoder(self, current_input):
        shapes = []
        for layer_i, n_output in enumerate(self.n_filters):
            shapes.append(current_input.get_shape().as_list())
            current_input = self.encode_block(current_input, self.filter_sizes[layer_i], layer_i, n_output)
            current_input = self.squeeze_excite_block_channel(current_input, n_output, 'ese_%s' %layer_i, ratio=1)
            current_input = self.squeeze_excite_block_temporal(current_input, current_input.get_shape().as_list()[1], 'ese_%s' %layer_i, ratio=4)
            current_input = tf.nn.dropout(current_input, self.keep_rate)
        return shapes, current_input

    def decoder(self, shapes, current_input):
        shapes.reverse()
        for layer_i, shape in enumerate(shapes):
            n_input = shape[2]
            print("layer %s: n_input %s" % (layer_i, n_input))
            if layer_i!=len(shapes)-1:
                current_input = self.decode_block(current_input, self.filter_sizes[layer_i], n_input, 2, tf.nn.relu, 'deconv_%s'%layer_i)
                current_input = self.squeeze_excite_block_channel(current_input, n_input, 'dse_%s' % layer_i, ratio=1)
                current_input = self.squeeze_excite_block_temporal(current_input, current_input.get_shape().as_list()[1], 'dse_%s' % layer_i, ratio=4)
            else:
                current_input = self.decode_block(current_input, self.filter_sizes[layer_i], n_input, 2, tf.nn.sigmoid,
                                                  'deconv_%s' % layer_i)
            current_input = tf.nn.dropout(current_input, self.keep_rate)
        y = tf.reshape(current_input, [-1, self.height*self.width, 1])
        return y

    def bottle_neck(self, current_input):
        dim0, dim1, dim2 = current_input.get_shape().as_list()
        hidden_dim = dim1*dim2
        mlp_layers = list(np.linspace(hidden_dim, self.image_featuredim, self.n_mlp+1, endpoint=True, dtype=int))
        print('hidden_dim', hidden_dim, self.image_featuredim, self.n_mlp, mlp_layers)

        current_input = tf.reshape(current_input, [-1, hidden_dim])
        for idx,output_dim in enumerate(mlp_layers[1:]):
            current_input = tf.layers.dense(current_input,
                                            output_dim,
                                              activation=tf.nn.tanh,
                                              trainable=True,
                                              kernel_regularizer=self.regularizer,
                                              name='neck_down%s' %idx)
            current_input = tf.nn.dropout(current_input, self.keep_rate)
        z = current_input
        mlp_layers.reverse()
        current_input_ =  current_input
        for idx, output_dim in enumerate(mlp_layers[1:]):
            current_input_ = tf.layers.dense(current_input_,
                                                output_dim,
                                                activation=tf.nn.tanh,
                                                trainable=True,
                                                kernel_regularizer=self.regularizer,
                                                name='neck_up%s' % idx)
            current_input_ = tf.nn.dropout(current_input_, self.keep_rate)
        current_input_ = tf.reshape(current_input_, [-1, dim1, dim2])
        return z, current_input_


    def cae(self, input_motifImage, train_phase, reuse=False,
                corruption=False):
        with tf.variable_scope('cae', reuse=reuse):
            input_motifImage = tf.reshape(input_motifImage, [-1, self.height*self.width, 1])
            shapes, current_input = self.encoder(input_motifImage)
            bottle_neck_vec, current_input = self.bottle_neck(current_input)
            current_input = self.decoder(shapes, current_input)
            return bottle_neck_vec, current_input

    def node2vec(self, central_image, pos_image, neg_image, train_phase):
        central_dim0, central_dim1, central_dim2, central_dim3 = tf.unstack(tf.shape(central_image))  # batch_size, num_pos, height, width
        central_embed, central_decode_input = self.cae(central_image, train_phase, reuse = False)
        central_embed = tf.reshape(central_embed, [-1, central_dim1, self.image_featuredim])
        # central_embed = tf.expand_dims(tf.reduce_mean(central_embed, axis=1), axis = 1)
        central_decode_input = tf.reshape(central_decode_input, [-1, central_dim1, central_dim2, central_dim3])
        print('central_dim3_embed', central_embed)

        pos_dim0, pos_dim1, pos_dim2, pos_dim3 = tf.unstack(tf.shape(pos_image))  # batch_size, num_pos, height, width
        pos_embed, pos_decode_input = self.cae(pos_image, train_phase, reuse = True)
        pos_embed = tf.reshape(pos_embed, [-1, pos_dim1, self.image_featuredim])
        # pos_embed = tf.expand_dims(tf.reduce_mean(pos_embed, axis=1), axis = 1)
        pos_decode_input = tf.reshape(pos_decode_input, [-1, pos_dim1, pos_dim2, pos_dim3])
        print('pos_dim3_embed', pos_embed)

        neg_dim0, neg_dim1, neg_dim2, neg_dim3 = tf.unstack(tf.shape(neg_image))  # batch_size, num_pos, height, width
        neg_embed, neg_decode_input = self.cae(neg_image, train_phase, reuse = True)
        neg_embed = tf.reshape(neg_embed, [-1, neg_dim1, self.image_featuredim])
        neg_decode_input = tf.reshape(neg_decode_input, [-1, neg_dim1, neg_dim2, neg_dim3])
        print('neg_embed', neg_embed)
        return central_embed, pos_embed, neg_embed, central_decode_input, pos_decode_input, neg_decode_input

    def temporal_attention(self, inputs, x, str_name):
        with tf.variable_scope('attention' + str_name):
            _,  sequence_length, hidden_size = tf.unstack(tf.shape(inputs))

            # Attention mechanism
            W_omega = tf.Variable(tf.random_normal([self.image_featuredim, self.attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, self.image_featuredim]), W_omega) + tf.reshape(b_omega, [1, -1]))
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
            exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

            output = tf.reduce_sum(x * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
            return output, alphas

    def aggregation(self, time_series_embed, date_embed, reuse):
        with tf.variable_scope('cae', reuse=reuse):
            train_date_embed = time_series_embed + date_embed
            output = train_date_embed
            for layer_id in range(2):
                with tf.variable_scope('self_att'+str(layer_id), reuse=False):
                    mult_head = MultiHeadedAttention(h = 4, d_model = self.image_featuredim)
                    output_ = mult_head.model(output, output, output)
                    output = output + batch_norm(output_,
                                                 center=True,
                                                 scale=True,
                                                 is_training=self.train_phase,
                                                 trainable=True,
                                                 scope='bn_layer1',
                                                 decay=0.9)
                    output_ = mult_head.PositionwiseFeedForward(output, self.image_featuredim, self.image_featuredim//2)
                    output = output + batch_norm(output_,
                                                 center=True,
                                                 scale=True,
                                                 is_training=self.train_phase,
                                                 trainable=True,
                                                 scope='bn_2',
                                                 decay=0.9)
            aggregate_embed, alpha = self.temporal_attention(output, time_series_embed, 'str_name')
        return aggregate_embed, alpha

    def pred_overall(self, central_embed, input_overall_label, n_class):
        label_mask = tf.greater(input_overall_label, -0.1)
        label_one_hot = tf.one_hot(tf.cast(tf.nn.relu(input_overall_label), tf.int32), n_class)
        logit_pred_label = tf.layers.dense(central_embed, n_class, activation=None,
                                  trainable=True,
                                  kernel_regularizer=self.regularizer,
                                  use_bias = True,
                                  name='pred_overall')
        pred_label = tf.nn.softmax(logit_pred_label)
        loss = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=logit_pred_label),label_mask))
        return pred_label, loss

    def pred_daily(self, central_embed, input_overall_label, n_class, reuse):
        with tf.variable_scope('pred_daily', reuse=reuse):
            label_mask = tf.greater(input_overall_label, -0.1)
            label_one_hot = tf.one_hot(tf.cast(tf.nn.relu(input_overall_label), tf.int32), n_class)
            logit_pred_label = tf.layers.dense(central_embed, n_class, activation=None,
                                      trainable=True,
                                      kernel_regularizer=self.regularizer,
                                      use_bias = True,
                                      name='pred_overall')
            pred_label = tf.nn.softmax(logit_pred_label)
            loss = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=logit_pred_label),label_mask))
        return pred_label, loss

    def model(self, decay = True):
        with tf.variable_scope(self.v_scope):
            central_motifImage = tf.placeholder(tf.float32, shape=[None, None, self.height, self.width])
            central_date = tf.placeholder(tf.int32, shape=[None, None])
            pos_motifImage = tf.placeholder(tf.float32, shape = [None, None, self.height, self.width])
            neg_motifImage = tf.placeholder(tf.float32, shape = [None, None, self.height, self.width])
            input_motifImage = tf.placeholder(tf.float32, shape = [None, self.height, self.width])
            input_overall_label = {}; input_daily_pos_label = {}; input_daily_neg_label = {}
            for cls in self.n_class:
                input_overall_label[cls] = tf.placeholder(tf.float32, shape=[None])
                input_daily_pos_label[cls] = tf.placeholder(tf.float32, shape=[None, None])
                input_daily_neg_label[cls] = tf.placeholder(tf.float32, shape=[None, None])

            self.keep_rate = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.date_embeddings = tf.get_variable('node_embeddings', [2000, self.image_featuredim],
                                              dtype=tf.float32,
                                              initializer=layers.xavier_initializer())
            self.date_embeddings = self.date_embeddings.assign(get_position_encoding(2000,self.image_featuredim))

            f_central_motifImage = convert_image_data_to_float(central_motifImage)
            f_pos_motifImage = convert_image_data_to_float(pos_motifImage)
            f_neg_motifImage = convert_image_data_to_float(neg_motifImage)
            f_input_motifImage = convert_image_data_to_float(input_motifImage)

            f_central_motifImage_mask = tf.greater(f_central_motifImage, 0)
            f_pos_motifImage_mask = tf.greater(f_pos_motifImage, 0)
            f_neg_motifImage_mask = tf.greater(f_neg_motifImage, 0)

            central_embed, pos_embed, neg_embed,\
            central_decode_input, pos_decode_input, neg_decode_input = self.node2vec(f_central_motifImage, f_pos_motifImage, f_neg_motifImage, self.train_phase)

            central_date_embed = tf.nn.embedding_lookup(self.date_embeddings, central_date)
            agg_central_embed, alpha = self.aggregation(central_embed, central_date_embed, reuse=False)
            print('agg_central_embed', agg_central_embed)

            pred_label = {}; pred_label_v = {}; loss_label = {}
            if self.overall_indicator:
                for cls in self.n_class:
                    with tf.variable_scope('pred' + cls, reuse=False):
                        pred_label[cls], loss_label[cls] = self.pred_overall(agg_central_embed, input_overall_label[cls], self.n_class[cls])
                        pred_label_v[cls] = tf.unstack(pred_label[cls], axis = -1)[1]
                        pred_label[cls] = tf.argmax(pred_label[cls], axis=-1)
            else:
                for cls in self.n_class:
                    with tf.variable_scope('pred' + cls, reuse=False):
                        pred_label[cls], loss_label[cls] = self.pred_daily(pos_embed,
                                                                           input_daily_pos_label[cls],
                                                                           self.n_class[cls], reuse = False)
                        neg_pred_label, neg_loss_label = self.pred_daily(neg_embed,
                                                                         input_daily_neg_label[cls],
                                                                         self.n_class[cls], reuse = True)
                        loss_label[cls] = loss_label[cls] + neg_loss_label
                        pred_label_v[cls] = tf.unstack(pred_label[cls], axis=-1)[1]
                        pred_label[cls] = tf.argmax(pred_label[cls], axis=-1)

            input_embed, _ = self.cae(f_input_motifImage, self.train_phase, reuse=True)

            params = tf.trainable_variables()
            regularizer = 0

            mse_loss = tf.maximum(0.0, tf.expand_dims(tf.reduce_sum(tf.expand_dims(agg_central_embed, axis = 1)*neg_embed, axis=2), axis = 1)
                                  - tf.expand_dims(tf.reduce_sum(tf.expand_dims(agg_central_embed, axis = 1)*pos_embed, axis=2), axis = 2) + 1)
            mse_loss = 0.1*tf.reduce_mean(tf.reduce_sum(mse_loss, axis = 1))
            mse_loss = mse_loss+tf.reduce_mean(tf.boolean_mask(tf.squared_difference(f_central_motifImage, central_decode_input), f_central_motifImage_mask))
            mse_loss = mse_loss+tf.reduce_mean(tf.boolean_mask(tf.squared_difference(f_pos_motifImage, pos_decode_input), f_pos_motifImage_mask))
            mse_loss = mse_loss+tf.reduce_mean(tf.boolean_mask(tf.squared_difference(f_neg_motifImage, neg_decode_input), f_neg_motifImage_mask))
            mse_loss = mse_loss+tf.reduce_mean(tf.stack(list(loss_label.values())))
            for p in params:
                regularizer += self.pr * tf.reduce_mean(tf.square(p))
            loss = mse_loss

            saver = tf.train.Saver(max_to_keep=1)

            if decay:
                opt = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(self.lr, self.global_step,
                                                                                      decay_steps=1000, decay_rate=0.99,
                                                                                      staircase=True))
            else:
                opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1, 1)

            gvs = opt.compute_gradients(loss)
            capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
            train_op = opt.apply_gradients(capped_gvs, global_step=self.global_step)

            init = tf.global_variables_initializer()
            return central_motifImage, central_date, pos_motifImage, neg_motifImage, input_motifImage, input_overall_label, \
                   input_daily_pos_label, input_daily_neg_label, self.keep_rate, self.train_phase, self.global_step, \
                   input_embed, pred_label, pred_label_v, agg_central_embed, alpha, loss, train_op, init, saver