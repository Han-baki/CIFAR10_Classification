#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

# def my_batch_norm(x, version = None):
#     """Batch normalization."""
# #     with tf.variable_scope(name):
#     params_shape = [x.get_shape()[-1]]

#     beta = tf.get_variable(
#       'beta', params_shape, tf.float32,
#       initializer=tf.constant_initializer(0.0, tf.float32))
#     gamma = tf.get_variable(
#       'gamma', params_shape, tf.float32,
#       initializer=tf.constant_initializer(1.0, tf.float32))
    
#     if version == 'Flatten':
#         mean, variance = tf.nn.moments(x, [0], name='moments')
#     else:        
#         mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

#     y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
#     y.set_shape(x.get_shape())
#     return y

def my_batch_norm(x, phase_train = True, version = None):
    """Batch normalization."""
#     with tf.variable_scope(name):
    params_shape = [x.get_shape()[-1]]

    beta = tf.get_variable(
      'beta', params_shape, tf.float32,
      initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(
      'gamma', params_shape, tf.float32,
      initializer=tf.constant_initializer(1.0, tf.float32))
    
    if version == 'Flatten':
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    else:        
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        
    ema = tf.train.ExponentialMovingAverage(decay = 0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda:(ema_mean, ema_var))
    y = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)
    y.set_shape(x.get_shape())
    return y

def _relu( x, leakiness=0.1):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def _stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.Conv2D."""
    return [1, stride, stride, 1]

def my_conv(name, x, filter_size, in_filters, out_filters, strides,initializer =None):
    """Convolution."""
    with tf.variable_scope(name):

        n = filter_size*filter_size*in_filters
        
        if initializer == 'zeros':
            kernel= tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.zeros_initializer)
        else:
            kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
        bias = tf.get_variable('DB',[out_filters], tf.float32, 
                               initializer = tf.constant_initializer(0.01))
        y = tf.nn.conv2d(x, kernel, _stride_arr(strides), padding='SAME')
        y = tf.nn.bias_add(y,bias)
    return y

def my_l2_regularizer(weight_decay_rate=0.0002, skip_name = ['gamma','beta']):
    result = []
    for var in tf.trainable_variables():
        
#         if var.op.name.find('gamma')>0 or var.op.name.find('beta')>0:
#             pass
#         else:
        
        in_skip_name = False
        for skname in skip_name:
            if var.op.name.find(skname)>0:
                in_skip_name = True
        if in_skip_name == True:
            pass
        else:
            result.append(tf.nn.l2_loss(var))
    return tf.multiply(weight_decay_rate, tf.reduce_sum(result))


def G_Conv2D(x, filter_size,in_filters, out_filters, strides=1, h_input='Z2', h_output='D4'):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input=h_input, h_output=h_output, 
        in_channels=in_filters, out_channels=out_filters, 
        ksize=filter_size)
    x_shape = x.get_shape().as_list()
    x_size_1, x_size_2 = x_shape[1], x_shape[2]
    n = x_size_1 * x_size_2 * in_filters
    if h_input =='D4':
        n = 8*n
    w = tf.get_variable('gconv_weight', w_shape, tf.float32,
                       initializer = tf.random_normal_initializer(stddev = np.sqrt(2.0/n)))
#     w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
    print(w_shape)
    if strides == 1:
        stride_arr = [1,1,1,1]
    elif strides ==2:
        stride_arr = [1,2,2,1]
    y = gconv2d(input=x, filter=w, strides=stride_arr, padding='SAME',
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info,use_cudnn_on_gpu = True)
    return y



def residual_block(x, in_filter, out_filter, strides, dropout = True, initializer = None, phase_train = True):
        
    with tf.variable_scope('conv1'):
        orig_x = x
        
        conv1 = my_batch_norm(x, phase_train)
#         conv1 = _relu(conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = my_conv('conv1', conv1, 3, in_filter, out_filter, strides=strides,initializer = initializer)
        
        if dropout:
            conv1 = tf.nn.dropout(conv1, keep_prob = 0.7)
    with tf.variable_scope('conv2'):
        conv2 = my_batch_norm(conv1, phase_train)
#         conv2 = _relu(conv2)
        conv2 = tf.nn.relu(conv2)
        conv2 = my_conv('conv2', conv2, 3, out_filter, out_filter, strides=1,initializer = initializer)
        

    if in_filter != out_filter:

        shortcut = my_conv('shortcut',orig_x, 1, in_filter, out_filter, strides = strides)
        shortcut = my_batch_norm(shortcut,phase_train)
        
#         pooled_input = tf.nn.avg_pool(orig_x, ksize=2, strides=2, padding='SAME')
#         shortcut = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0],
#                                              [(out_filter-in_filter) // 2,(out_filter-in_filter) // 2]])
        
    else:
        shortcut = orig_x

    output = conv2 + shortcut
    return output

def G_residual_block(x, in_filter, out_filter, strides, dropout = True, phase_train = True):
        
    with tf.variable_scope('conv1'):
        orig_x = x
        conv1 = my_batch_norm(x, phase_train)
#         conv1 = _relu(conv1)
        conv1 = tf.nn.relu(conv1)
        conv1 = G_Conv2D(conv1, 3, in_filter, out_filter, strides = strides, h_input='D4', h_output='D4')
#         print('conv1',conv1.get_shape().as_list())
        if dropout:
            conv1 = tf.nn.dropout(conv1, keep_prob = 0.7)
    with tf.variable_scope('conv2'):
        conv2 = my_batch_norm(conv1, phase_train)
#         conv2 = _relu(conv2)
        conv2 = tf.nn.relu(conv2)
        conv2 = G_Conv2D(conv2, 3, out_filter, out_filter, strides = 1, h_input='D4', h_output='D4')
#         print('conv2',conv2.get_shape().as_list())
    if in_filter != out_filter:
        pooled_input = tf.nn.avg_pool(orig_x, ksize=2, strides=2, padding='SAME')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0],
                                             [8*(out_filter-in_filter) // 2,8*(out_filter-in_filter) // 2]])
#         print('x',x.get_shape().as_list())
#         print('pad',padded_input.get_shape().as_list())
    else:
        padded_input = orig_x

    output = conv2 + padded_input
    return output

