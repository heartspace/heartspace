import tensorflow as tf
import numpy as np

def mask(images, thres = 10):#mask based on input
    thres = 10
    cond = tf.greater(images, thres)
    return cond

def cutomized_mask(images):#mask based on input
    mask = tf.sparse_to_dense([[1,2], [3,4]], tf.shape(images), [1, 1])
    return mask

def sequential_mask(first_position,length, image):#mask based on input
    image_shape = image.get_shape().as_list()
    context_a = tf.ones(first_position)
    mask = tf.zeros(length)
    context_b = tf.ones(tf.reduce_prod(image_shape)-first_position-length)
    mask = tf.concat([context_a, mask, context_b], axis = 0)
    mask = tf.reshape(mask, image_shape)
    mask = tf.cast(mask, tf.bool)
    return mask

def comb_mask(first_position,length, image):
    mask1 = mask(image, thres=10)
    mask2 = sequential_mask(first_position,length, image)
    return tf.logical_and(mask1, mask2)

if __name__ == '__main__':
    vocab_size = 10
    embedding_dim = 10
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W").assign(np.random.randint(0, high=20, size=[vocab_size, embedding_dim], dtype='l'))
    with tf.Session() as sess:
        W_, mask_, mask1_ = sess.run([W, mask(W), cutomized_mask(W)])
        print(W_, mask_, mask1_)
        mask = sess.run(comb_mask(15, 23, W))
        print(mask)