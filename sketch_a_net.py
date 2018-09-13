import tensorflow as tf
import tflearn
import numpy
import os 
from data_layer import load_pretrained_model

def weights_variable(shape, weights=None):
    """
    Args:
        shape: the shape of the fliter varibale to be created
        weights:pretrained weights

    Returns:
        tf.Variable with the respectives weights
    """
    if weights is not None and weights.shape != shape:
        raise ValueError("The pretrained model\'s shapes don\'t match with the layer shapes") 
    initial = tf.truncated_normal(shape, stddev=0.1) if weights is None else weights
    return tf.Variable(initial, name='weights')

def biases_variable(shape, biases=None):
    
    if biases is not None and biases.shape != shape:
        raise ValueError("The pretrained is not matched the shape")
    initial = tf.truncated_normal(shape, stddev=0.1) if biases is None else biases
    return tf.Variable(initial, name='biases')

def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.scalar(tensor_name+'/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(tensor_name + '/activations', x)

def inference(images,dropout_prob=1.0,pretrained=(None,None), visualize=False):
    """
    This prepares the tensorflow graph for the vanilla Sketch-A-Net network
    and returns the tensorflow op from the last fully_connected layer
    
    Args:
    images: the input images of shape(N, H, W, C) for the network from the data layer
    Returns: Logits for the softmax loss
    
    """
    weights, biases = pretrained
    #print(111111)
    with tf.name_scope('L1') as scope:
        weights1 = weights_variable((15,15,6,64), None if weights is None else weights['conv1'])
        biases1 = biases_variable((64,), None if biases is None else biases['conv1'])
        conv1 = tf.nn.conv2d(images, weights1, [1,3,3,1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID',name='pool1')

    with tf.name_scope('L2') as scope:
        weights2 = weights_variable((5,5,64,128), None if weights is None else weights['conv2'])
        biases2 = biases_variable((128,), None if biases is None else biases['conv2'])
        conv2 = tf.nn.conv2d(pool1, weights2, [1,1,1,1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')

    with tf.name_scope('L3') as scope:
        weights3 = weights_variable((3,3,128,256), None if weights is None else weights['conv3'])
        biases3 = biases_variable((256, ), None if biases is None else biases['conv3'])
        conv3 = tf.nn.conv2d(pool2, weights3, [1,1,1,1], padding='SAME', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3),name='relu3')
        
    with tf.name_scope('L4') as scope:
        weights4 = weights_variable((3,3,256,256), None if weights is None else weights['conv4'])
        biases4 = biases_variable((256,), None if biases is None else biases['conv4'])
        conv4 = tf.nn.conv2d(relu3, weights4, [1,1,1,1], padding='SAME', name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4), name='relu4')

    with tf.name_scope('L5') as scope:
        weights5 = weights_variable((3,3,256,256), None if weights is None else weights['conv5'])
        biases5 = biases_variable((256,), None if biases is None else biases['conv5'])
        conv5 = tf.nn.conv2d(relu4, weights5, [1,1,1,1], padding='SAME', name='conv5')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5), name='relu5')
        pool5 = tf.nn.max_pool(relu5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
    
    with tf.name_scope('L6') as scope:
        weights6 = weights_variable((7,7,256,512), None if weights is None else weights['conv6'])
        biases6 = biases_variable((512,), None if biases is None else biases['conv6'])
        fc6 = tf.nn.conv2d(pool5, weights6, [1,1,1,1], padding='VALID', name='fc6')
        relu6 = tf.nn.relu(tf.nn.bias_add(fc6, biases6), name='relu6')
        dropout6 = tf.nn.dropout(relu6, keep_prob=dropout_prob, name='dropout6')
    
    with tf.name_scope('L7') as scope:
        weights7 = weights_variable((1,1,512,512), None if weights is None else weights['conv7'])
        biases7 = biases_variable((512,), None if biases is None else biases['conv7'])
        fc7 = tf.nn.conv2d(dropout6, weights7, [1,1,1,1], padding='VALID', name='fc7')
        relu7 = tf.nn.relu(tf.nn.bias_add(fc7, biases7),name='relu7')
        dropout7 = tf.nn.dropout(relu7, keep_prob=dropout_prob, name='dropout7')

    with tf.name_scope('L8') as scope:
        weights8 = weights_variable((1,1,512,250), None if weights is None else weights['conv8'])
        biases8 = biases_variable((250, ), None if biases is None else biases['conv8'])
        fc8 = tf.nn.conv2d(dropout7, weights8, [1,1,1,1], padding='VALID', name='fc8')
    logits = tf.reshape(tf.nn.bias_add(fc8, biases8), [-1,250])
    if visualize:
        activation = {
            'relu1':relu1,
            'relu2':relu2,
            'relu3':relu3,
            'relu4':relu4,
            'relu5':relu5,
            'relu6':relu6,
            'relu7':relu7}
        return logits, activation
    return logits
def loss(logits, labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
        name='entropy')
    entropy_mean = tf.reduce_mean(cross_entropy, name='entropy_mean')
    tf.summary.scalar('loss', entropy_mean)
    return entropy_mean

def evaluation(logits, labels, k, is_train):
    
    if not is_train:
        logits = tf.reduce_sum(tf.reshape(logits,[10,-1,250]),axis=0)
    correct = tf.nn.in_top_k(logits, tf.cast(labels[:tf.shape(logits)[0]], tf.int32),k)
    return prediction


def training(loss, lr, global_step, decay_step=100, decay_rate=0.96, staircase=True):
    learnning_rate = tf.train.exponential_decay(lr, global_step, decay_step, decay_rate, staircase, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('learning_rate', learning_rate)
    return train_op

    































     
    













