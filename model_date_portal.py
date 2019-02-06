import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
# from kdd_model.node2vec_image import NN
from heartspace import heartspaceNN
import os
import collections


class HRAlgorithm:


    def __init__(self, num_pos, num_neg, featureDimension, penalty_rate,
                 v_scope, keep_rate, obv=10, batch_size = 2, learning_rate = 1e-2,
                 attention_size = 32, n_filters = [4, 16], filter_sizes= [5, 5],
                 image_featuredim = 100, n_mlp = 1, NN = heartspaceNN, num_input = 1, overall_indicator = 1, n_class = {}):  # n is number of users
        self.motifs = {}
        self.batch_size = batch_size
        self.obv = obv
        self.lr = learning_rate
        self.total_loss = 0
        self.reward = 0
        self.WeightReward = 0
        self.userRecord = {}

        self.kr = keep_rate
        self.pr = penalty_rate
        self.v_scope = v_scope

        self.NN = NN(num_pos=num_pos,
                  num_neg=num_neg,
                  featureDimension=featureDimension,
                  penalty_rate=penalty_rate,
                  learning_rate=self.lr,
                  batch_size=batch_size,
                  v_scope=v_scope,
                  attention_size=attention_size,
                  n_filters = n_filters,
                  filter_sizes = filter_sizes,
                  image_featuredim = image_featuredim,
                  n_mlp = n_mlp,
                  num_input = num_input,
                  overall_indicator = overall_indicator,
                  n_class = n_class)

        self.central_motifImage, self.central_date, self.pos_motifImage, self.neg_motifImage, \
        self.input_motifImage, self.input_overall_label, self.input_daily_pos_label, \
        self.input_daily_neg_label, self.keep_rate, self.train_phase, self.global_step, \
        self.central_embed, self.pred, self.pred_label_v, self.agg_central_embed, self.alpha,\
        self.loss, self.train_op, self.init, self.saver = self.NN.model()

        self.batch_central_motifImage = []
        self.batch_pos_motifImage = []
        self.batch_neg_motifImage = []


    def create(self, pretrain_flag=0, save_file=''):
        graph = tf.Graph().as_default()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.log_device_placement = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        print('pretrain_flag', pretrain_flag)
        if pretrain_flag == 0:
            self.sess.run(self.init)
        else:
            self.sess.run(self.init)
            print('restore from %s' % save_file)
            self.saver.restore(self.sess, save_file + '.ckpt')
        print('create', 'pretrain_flag', pretrain_flag)

    def save_weight(self, save_file):
        try:
            self.saver.save(self.sess, save_file+'.ckpt')
        except:
            print('need to makedir')
            os.makedirs(save_file, exist_ok=True)
            self.saver.save(self.sess, save_file+'.ckpt')
        print("Save model to file as pretrain.")
        
    def feedbatch(self, batch_central_motifImage, batch_central_date, batch_pos_motifImage, batch_neg_motifImage):
        feed_dict = {self.central_motifImage: batch_central_motifImage, self.central_date: batch_central_date, self.pos_motifImage: batch_pos_motifImage,
                     self.neg_motifImage: batch_neg_motifImage,
                     self.keep_rate: self.kr, self.train_phase: 1}
        _, step, loss, pred = self.sess.run([self.train_op, self.global_step, self.loss, self.pred], feed_dict)
        return (step, loss)

    def feedbatch_daily(self, batch_central_motifImage, batch_central_date, batch_pos_motifImage, batch_neg_motifImage, batch_pos_label, batch_neg_label):
        feed_dict = {self.central_motifImage: batch_central_motifImage, self.central_date: batch_central_date, self.pos_motifImage: batch_pos_motifImage,
                     self.neg_motifImage: batch_neg_motifImage,
                     self.keep_rate: self.kr, self.train_phase: 1}
        for cls in batch_pos_label:
            feed_dict[self.input_daily_pos_label[cls]] = batch_pos_label[cls]
        for cls in batch_neg_label:
            feed_dict[self.input_daily_neg_label[cls]] = batch_neg_label[cls]
        _, step, loss = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict)
        return (step, loss)

    def feedbatch_overall(self, batch_central_motifImage, batch_central_date, batch_pos_motifImage, batch_neg_motifImage, batch_overall_label):
        feed_dict = {self.central_motifImage: batch_central_motifImage, self.central_date: batch_central_date, self.pos_motifImage: batch_pos_motifImage,
                     self.neg_motifImage: batch_neg_motifImage,
                     self.keep_rate: self.kr, self.train_phase: 1}
        for cls in batch_overall_label:
            feed_dict[self.input_overall_label[cls]] = batch_overall_label[cls]
        _, step, loss, pred = self.sess.run([self.train_op, self.global_step, self.loss, self.pred], feed_dict)
        return (step, loss)

    def getDailyEmbed(self, motifImage):
        return self.sess.run(self.central_embed, feed_dict={self.input_motifImage: motifImage,
                                                          self.train_phase: 0, self.keep_rate: 1.0})

    def getOverallEmbed(self, motifImage, input_date):
        return self.sess.run([self.agg_central_embed, self.alpha], feed_dict={self.central_motifImage: motifImage,
                                                                self.central_date: input_date,
                                                            self.train_phase: 0, self.keep_rate: 1.0})

    def getDailyLabel(self, motifImage):
        return self.sess.run([self.pred, self.pred_label_v], feed_dict={self.pos_motifImage: motifImage,
                                                          self.train_phase: 0, self.keep_rate: 1.0})

    def getOverallLabel(self, motifImage, input_date):
        return self.sess.run([self.pred, self.pred_label_v], feed_dict={self.central_motifImage: motifImage,
                                                                self.central_date: input_date,
                                                            self.train_phase: 0, self.keep_rate: 1.0})
    
if __name__ == '__main__':
    featureDimension = [24, 60]; embed_size = 32; penalty_rate = 0
    v_scope = ''; sampled_true = 1
    sampled_false = 4; keep_rate = 1.0; batch_size = 32; learning_rate = 1e-4
    num_pos = 1; num_neg = 4; num_input = 3; batch_size = 2

    alg = HRAlgorithm(num_pos, num_neg, featureDimension=featureDimension, penalty_rate=0,
                       v_scope='', keep_rate=0.9, obv=50, batch_size=batch_size, learning_rate=learning_rate,
                       attention_size=32, n_filters=[32, 64, 64, 128, 128], filter_sizes=[9, 7, 7, 5, 5],
                       image_featuredim=embed_size, n_mlp=1, NN=heartspaceNN, num_input=num_input,
                       overall_indicator=1, n_class={})

    alg.create(pretrain_flag=0, save_file='')

    avg_loss = 0

    for i in range(200):
        'batch input data'
        centralImage = np.random.randint(0, high=200, size = (batch_size, num_input, featureDimension[0],featureDimension[1]))
        posImage = np.random.randint(0, high=200, size = (batch_size, num_pos, featureDimension[0],featureDimension[1]))
        negImage = np.random.randint(0, high=200, size=(batch_size, num_neg, featureDimension[0], featureDimension[1]))
        centralDate = np.random.randint(0, high=50, size = (batch_size, num_input))
        'update alg'
        step, loss_ = alg.feedbatch(centralImage, centralDate, posImage, negImage)
        avg_loss += loss_
        if step % 100 == 0:
            avg_loss /= 100
            print('step:%s, loss:%.4f' % (step, avg_loss))
            avg_loss = 0

    'obtain embeddings'
    agg_embed, agg_weight = alg.getOverallEmbed([centralImage[0]], [centralDate[0]])
    daily_embed = alg.getDailyEmbed(centralImage[0])


