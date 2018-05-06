# coding=utf-8


from  __future__ import division

import tensorflow as tf
from tensorflow.examples.tutorials.cifar10 import cifar10, cifar10_input
import  numpy as np
import time


STEPS = 10000
batch_size =128

data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'



parameters = {
    'w1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=1e-1), name='w1'),
    'w2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='w2'),
    'w3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='w3'),
    'w4': tf.Variable(tf.truncated_normal([3, 3, 64, 256], dtype=tf.float32, stddev=1e-1), name='w4'),
    'w5': tf.Variable(tf.truncated_normal([3, 3, 256, 128], dtype=tf.float32, stddev=1e-1), name='w5'),
    'fc1': tf.Variable(tf.truncated_normal([128*24*24, 1024], dtype=tf.float32, stddev=1e-2), name='fc1'),
    'fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=1e-2), name='fc2'),
    'softmax': tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-2), name='fc3'),
    'bw1': tf.Variable(tf.random_normal([32])),
    'bw2': tf.Variable(tf.random_normal([64])),
    'bw3': tf.Variable(tf.random_normal([64])),
    'bw4': tf.Variable(tf.random_normal([256])),
    'bw5': tf.Variable(tf.random_normal([128])),
    'bc1': tf.Variable(tf.random_normal([1024])),
    'bc2': tf.Variable(tf.random_normal([1024])),
    'bs': tf.Variable(tf.random_normal([10]))
}

def conv2d(_x, _w, _b):
    '''
         封装的生成卷积层的函数
         因为NNIST的图片较小,这里采用1,1的步长
    :param _x:  输入
    :param _w:  卷积核
    :param _b:  bias
    :return:    卷积操作
    '''
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b))

def lrn(_x):
    '''
    作局部响应归一化处理
    :param _x:
    :return:
    '''
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def max_pool(_x, f):
    '''
        最大池化处理,因为输入图片尺寸较小,这里取步长固定为1,1,1,1
    :param _x:
    :param f:
    :return:
    '''
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')

def loss(logits, labels):
    '''
    使用tf.nn.sparse_softmax_cross_entropy_with_logits将softmax和cross_entropy_loss计算合在一起
    并计算cross_entropy的均值添加到losses集合.以便于后面输出所有losses
    :param logits:
    :param labels:
    :return:
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def inference(_parameters,_dropout):
    '''
     定义网络结构和训练过程
    :param _parameters:  网络结构参数
    :param _dropout:     dropout层的keep_prob
    :return:
    '''

    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])


    # 第一卷积层
    conv1 = conv2d(image_holder, _parameters['w1'], _parameters['bw1'])
    lrn1 = lrn(conv1)
    pool1 = max_pool(lrn1, 2)

    # 第二卷积层
    conv2 = conv2d(pool1, _parameters['w2'], _parameters['bw2'])
    lrn2 = lrn(conv2)
    pool2 = max_pool(lrn2, 2)

    # 第三卷积层
    conv3 = conv2d(pool2, _parameters['w3'], _parameters['bw3'])

    # 第四卷积层
    conv4 = conv2d(conv3, _parameters['w4'], _parameters['bw4'])

    # 第五卷积层
    conv5 = conv2d(conv4, _parameters['w5'], _parameters['bw5'])
    pool5 = max_pool(conv5, 2)

    # FC1层
    shape = pool5.get_shape() # 获取第五卷基层输出结构,并展开
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, _parameters['fc1']) + _parameters['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=_dropout)

    # FC2层
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, _parameters['fc2']) + _parameters['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=_dropout)

    # softmax层
    logits = tf.add(tf.matmul(fc2_drop, _parameters['softmax']),_parameters['bs'])
    losses = loss(logits, label_holder)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(losses)

    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)


    # 创建默认session,初始化变量
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 启动图片增强线程队列
    tf.train.start_queue_runners()

    initop = tf.global_variables_initializer()
    sess.run(initop)

    for step in range(STEPS):
        batch_xs, batch_ys = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, losses], feed_dict={image_holder: batch_xs,
                                                                label_holder: batch_ys} )
        if step % 20 == 0:
            print('step:%5d. --lost:%.6f. '%(step, loss_value))
    print('train over!')

    num_examples = 10000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size  # 除去不够一个batch的
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                      label_holder: label_batch})
        true_count += np.sum(predictions)  # 利用top_k_op计算输出结果
        step += 1

    precision = true_count / total_sample_count

    print('precision @ 1=%.3f' % precision)


if __name__ == '__main__':
    start = time.clock() #计算开始时间
    
    cifar10.maybe_download_and_extract()
    inference(parameters, 0.7)
    
    end = time.clock() #计算程序结束时间
    print(("running time is %g s") % (end-start))