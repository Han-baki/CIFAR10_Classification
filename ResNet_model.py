#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from layers_and_blocks import my_l2_regularizer, my_conv, residual_block
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class ResNet(): 
    def __init__(self, number_n = 3, input_shape=[32,32,3], num_classes=10, filter_num = 16,
                weight_decay = 0.0001, dropout = False):
        
        
        self.number_n = number_n
        self.num_classes = num_classes
        self.filter_num = filter_num
        self.dropout = dropout
#         with tf.Session() as sess:
#             print(tf.trainable_variables())
        self.phase_train = tf.placeholder(tf.bool)
        self.logits = self.model(input_shape, self.phase_train)
        self.outputs = tf.nn.softmax(self.logits)
        self.reg_loss = my_l2_regularizer(weight_decay)
#         with tf.Session() as sess:
#             print(tf.trainable_variables())
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits = self.logits, labels = self.labels_one_hot))
        self.loss = self.c_loss + self.reg_loss
      
        
    def model(self, input_shape, phase_train):
      
        self.input = tf.placeholder(tf.float32, shape = [None]+input_shape)
        self.labels = tf.placeholder(tf.int32, shape = [None])
        self.labels_one_hot = tf.one_hot(self.labels,self.num_classes)
        f_num = self.filter_num
        
        
        with tf.variable_scope('init'):
            x = self.input
#             x = Conv2D(filters = f_num ,kernel_size = 3, strides = 1, padding = 'SAME')(x)
            x = my_conv('init_conv', x, 3, 3, f_num, strides = 1)
            

        with tf.variable_scope('unit_1_0'):
            x = residual_block(x, f_num, f_num, strides = 1, dropout=self.dropout, phase_train = phase_train)
        for i in range(1,self.number_n):
            with tf.variable_scope('unit_1_%d' % i):
                x = residual_block(x, in_filter = f_num, out_filter = f_num, strides = 1, dropout=self.dropout, phase_train = phase_train)
            
        with tf.variable_scope('unit_2_0'):
            x = residual_block(x, f_num, f_num*2, strides = 2, dropout=self.dropout, phase_train = phase_train)
        for i in range(1,self.number_n):
            with tf.variable_scope('unit_2_%d' % i):
                x = residual_block(x, in_filter = f_num*2, out_filter = f_num*2, strides = 1, dropout=self.dropout, phase_train = phase_train)
        
        with tf.variable_scope('unit_3_0'):
            x = residual_block(x, f_num*2, f_num*4, strides = 2, dropout=self.dropout, phase_train = phase_train)
        for i in range(1,self.number_n):
            with tf.variable_scope('unit_3_%d' % i):
                x = residual_block(x, in_filter = f_num*4, out_filter = f_num*4, strides = 1, dropout=self.dropout, phase_train = phase_train)
          
        with tf.variable_scope('unit_last'):

            x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
#             print(x.get_shape().as_list())
            
        with tf.variable_scope('logit'):
            logit = Dense(self.num_classes)(x)
        
        return logit
        
    def train(self,optimizer):
        
        self.training = optimizer

