#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from layers_and_blocks import my_l2_regularizer, my_batch_norm, my_conv
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class BasicCNN(): 
    def __init__(self,number_n=0, input_shape=[32,32,3], num_classes=10, 
                 weight_decay = 0.0001, 
                 filter_num = 32, dropout = True,
                 res_initializer = 'zeros'):
        
        self.number_n = number_n
        self.num_classes = num_classes
        self.filter_num = filter_num
        self.dropout = dropout
        self.res_initializer= res_initializer
        
        self.phase_train = tf.placeholder(tf.bool)#수정
        self.logits = self.model(input_shape, self.phase_train)#수정
        self.outputs = tf.nn.softmax(self.logits)
        self.reg_loss = my_l2_regularizer(weight_decay, skip_name =['gamma','beta','res'])
#         with tf.Session() as sess:
#             print(tf.trainable_variables())
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits = self.logits, labels = self.labels_one_hot))
        self.loss = self.c_loss + self.reg_loss
        
    
    def model(self, input_shape, phase_train):#수정
      
        self.input = tf.placeholder(tf.float32, shape = [None]+input_shape)
        self.labels = tf.placeholder(tf.int32, shape = [None])
        self.labels_one_hot = tf.one_hot(self.labels,self.num_classes)
        f_num = self.filter_num
        
        x = self.input
        with tf.variable_scope('block1_1'):
            
            x = my_batch_norm(x,phase_train)
            x = my_conv('basic_conv1', x, 3, 3, f_num, strides = 1)
            x = tf.nn.relu(x)
            
        with tf.variable_scope('block1_2'):
            x = my_batch_norm(x,phase_train)
            x = my_conv('basic_conv2', x, 3, f_num, f_num, strides = 1)
            x = tf.nn.relu(x)
            
        if self.number_n>0:
            for i in range(self.number_n):
                name = 'res1_'+str(i)
                with tf.variable_scope(name):            
                    x = residual_block(x, f_num, f_num,strides = 1, dropout = self.dropout, initializer =self.res_initializer)

        x = MaxPooling2D([2,2],[2,2],padding = 'SAME')(x)
        if self.dropout:            
            x = Dropout(rate = 0.2)(x)
        
        with tf.variable_scope('block2_1'):
            
            x = my_batch_norm(x,phase_train)
            x = my_conv('basic_conv1', x, 3, f_num, 2*f_num, strides = 1)
            x = tf.nn.relu(x)
            
        with tf.variable_scope('block2_2'):
            
            x = my_batch_norm(x, phase_train)
            x = my_conv('basic_conv2', x, 3, 2*f_num, 2*f_num, strides = 1)
            x = tf.nn.relu(x)
            
        if self.number_n>0:
            for i in range(self.number_n):
                name = 'res2_'+str(i)
                with tf.variable_scope(name):            
                    x = residual_block(x, 2*f_num, 2*f_num,strides = 1, dropout = self.dropout, initializer =self.res_initializer)

        x = MaxPooling2D([2,2],[2,2],padding = 'SAME')(x)
        if self.dropout:            
            x = Dropout(rate = 0.3)(x)
        
        with tf.variable_scope('block3_1'):
            
            x = my_batch_norm(x, phase_train)
            x = my_conv('basic_conv1', x, 3, 2*f_num, 4*f_num, strides = 1)
            x = tf.nn.relu(x)
            
        with tf.variable_scope('block3_2'):
            
            x = my_batch_norm(x, phase_train)
            x = my_conv('basic_conv2', x, 3, 4*f_num, 4*f_num, strides = 1)
            x = tf.nn.relu(x)
            
        if self.number_n>0:
            for i in range(self.number_n):
                name = 'res3_'+str(i)
                with tf.variable_scope(name):            
                    x = residual_block(x, 4*f_num, 4*f_num,strides = 1, dropout = self.dropout, initializer =self.res_initializer)
            
        x = MaxPooling2D([2,2],[2,2],padding = 'SAME')(x)
        if self.dropout:            
            x = Dropout(rate = 0.4)(x)
        
        with tf.variable_scope('flat1'):
            
            x = Flatten()(x) 
            x = my_batch_norm(x, phase_train,version= 'Flatten')
            x = Dense(256)(x)
            x = tf.nn.relu(x)
            if self.dropout:
                x = Dropout(rate= 0.5)(x)
        
        with tf.variable_scope('flat2'):
            
            x = my_batch_norm(x, phase_train,version= 'Flatten')
            logit = Dense(self.num_classes, activation = None)(x) 
            
        return logit
        
        
    def train(self,optimizer):
        
        self.training = optimizer

