import tensorflow as tf
import numpy as np
from data_layer import load_pretrained_model,DataLayer
import sketch_a_net as sn
import sys
import argparse
import os

import time

FLAGS = None
def do_eval(sess, eval_correct, images_placeholder, dataset, is_train):
    print('Running Evaluation on'+('train' if is_train else 'test')+ 'set')
    num_examples = (DataLayer.NUM_TRAIN_PER_CLASS if is_train else DataLayer.NUM_TEST_PER_CLASS) * DataLyer.NUM_CLASS
    steps_per_epoch = num_examples // (dataset.train_batch_size if is_train else dataset.test_batch_size)
    last_step_size = num_examples % (dataset.train_batch_size if is_train else dataset.test_batch_size)
    batch_size = (dataset.train_batch_size if is_train else dataset.test_batch_size))

    true_count = 0
    strat_time = time.time()

    for step in range(steps_per_epoch):
        images, labels = dataset.next_batch_train() if is_train else dataset.next_batch_test()
        count = sess.run(eval_correct, feed_dice={
            images_placeholder:images,
            labels_placeholder:labels
        })
        true_count += true_count
        precison = float(count)/batch_size
        print('Num examples : %d Num Correct: %d Precision : %d' % (batch_size, count, precision))









def main(_a):
    pretrained = load_pretrained_model(FLAGS.model_path if FLAGS.no_pretrain else None)
    #print(len(pretrained))
    dataset = DataLayer(FLAGS.data_path, batch_size=FLAGS.batch_size)
    num_test = 6500
    num_steps= 6500 // 54
    fprediction = 0
    sess =tf.Session()
        #test_images, test_labels=dataset.next_batch_test(54)
        #print(test_images.shape,labels.shape)

    images = tf.placeholder(tf.float32,shape=(None, 225,225,6))
    labels = tf.placeholder(tf.float32, shape=(None,))
    dr = tf.placeholder_with_default(1.0,shape=())
    logits = sn.inference(images,dr,pretrained=pretrained)
    prediction = sn.evaluation(logits, labels,1,False)
    init = tf.global_variables_initializer()
           
    sess.run(init)
    for i in range(num_steps):
        test_images, test_labels=dataset.next_batch_test(54) 
        c_logits = sess.run(logits, feed_dict={images:test_images, dr:FLAGS.dr})
           #print(type(c_logits))
        fprediction = fprediction + sess.run(prediction, feed_dict={logits:c_logits,labels:test_labels})/float(num_steps)
           #print(prediction)
                
    print(fprediction) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--logdir',
            type=str,
            default='/tmp/tensorflow/sketch-a-net/',
            help='Directory to save the log'
            )
    parser.add_argument(
            '--model_path',
            type=str,
            default='SketchANetModel/model_with_order_info_256.mat',
            help='the .mat file with the pretrained weights download from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
            )
    parser.add_argument(
            '--data_path',
            default='sketch-dataset/dataset_with_order_info_256.mat',
            help='The .mat file with the dataset downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
            )
    parser.add_argument(
            '--lr',
            default='0.001',
            help='the initial learning rate for the optimizer'
            )
    parser.add_argument(
            '--decay_step',
            default=250,
            help='the decay steps for the exponential decay learning rate'
            )
    parser.add_argument(
            '--decay_rate',
            default=0.96,
            help='the decay rate for exponential decay learning rate'
            )

    parser.add_argument(
            '--dr',
            type=float,
            default=0.5,
            help='the probability to keep a neuron in dropout'
            )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=54,
            help='bathch_size'
            )
    parser.add_argument(
            '--epoch',
            type=int,
            default=500,
            )
    parser.add_argument(
            '--topk',
            type=int,
            default=1
            )

    parser.add_argument(
            '--eval_only',
            action='store_false',
            default=True
            )
    parser.add_argument(
            '--no_pretrain',
            action='store_false',
            default=True
            )
    parser.add_argument(
            '--reset_step',
            action='store_true',
            default=False
            )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
