#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm_notebook

def load_cifar10(path = 'cifardata/cifar-10-batches-py/'):
  
    import os
    import pickle
    import numpy as np

    def unpickle(file):
        import pickle
        with open(file,'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    def load_data(path, file):
        
        return unpickle(path + file)
      
    cifar_data_1 = load_data(path,'data_batch_1')
    cifar_data_2 = load_data(path,'data_batch_2')
    cifar_data_3 = load_data(path,'data_batch_3')
    cifar_data_4 = load_data(path,'data_batch_4')
    cifar_data_5 = load_data(path,'data_batch_5')
    cifar_test = load_data(path,'test_batch')
    cifar_data = [cifar_data_1,cifar_data_2,cifar_data_3,cifar_data_4,cifar_data_5,cifar_test]
    datas = []
    labels = []

    for i in range(6):
        datas.append(cifar_data[i][b'data'])
        labels.append(cifar_data[i][b'labels'])

    input_datas = []
    input_labels = []

    for i in range(6):
        input_datas.append(np.transpose(np.reshape(datas[i],[-1,3,32,32]), [0,2,3,1]))
        input_labels.append(np.array(labels[i]))
    
    print('image shape: ', np.shape(input_datas))
    print('label shape: ', np.shape(input_labels))
    return np.array(input_datas), np.array(input_labels)

  
    

def mean_std_normalization_per_pixels(train_datas, test_datas, print_mean_std = False):
    
    mean = np.mean(train_datas, axis = (0,3))
    std = np.std(train_datas, axis = (0,3))
    if print_mean_std:        
        print('old train mean: ',mean)
        print('old train std: ',std)

    train_datas = train_datas.astype(float) - mean[np.newaxis,:,:,np.newaxis]
    train_datas /= std[np.newaxis, :,:, np.newaxis]
    test_datas = test_datas.astype(float) - mean[np.newaxis,:,:,np.newaxis]
    test_datas /= std[np.newaxis, :,:, np.newaxis]
    
    if print_mean_std:
        print('new train mean:',np.mean(train_datas, axis = (0,3)))
        print('new train std:',np.std(train_datas, axis = (0,3)))
        print('new test mean:',np.mean(test_datas, axis = (0,3)))
        print('new test std:',np.std(test_datas, axis = (0,3)))
              
    return train_datas, test_datas

              

def data_generate(image_input_datas, batch_size):
    
     
    padheight = 4
    padwidth = 4
    paddings = tf.constant([[0,0],[padheight,padheight],[padwidth,padwidth],[0,0]])

    result_imgs = tf.pad(image_input_datas, paddings)

    result_imgs = tf.random_crop(result_imgs, [batch_size,32, 32, 3])

    result_imgs = tf.image.random_flip_left_right(result_imgs)

    result_imgs = tf.image.random_brightness(result_imgs,max_delta=0.25)

    result_imgs = tf.image.random_contrast(result_imgs,lower=0.2, upper=1.8)
    
    return result_imgs


class cifar10_compile():
    
    def __init__(self, model, Session, train_datas, train_labels, 
                  test_datas, test_labels, 
                  batch_size = 100, 
                  ):
              
        self.model = model
        self.sess = Session
        self.train_datas = train_datas
        self.train_labels = train_labels
        self.test_datas = test_datas
        self.test_labels = test_labels
        
        self.train_acc_list = []
        self.test_acc_list = []
              
        self.optimizer = model.training
        self.loss = model.loss
        
        self.batch_size = batch_size
        self.total_batch = int(len(train_datas)/batch_size)

        self.image_input_datas = tf.placeholder(tf.float32,shape = [batch_size,32,32,3])
        self.data_aug_imgs = data_generate(self.image_input_datas, batch_size = batch_size)
  
    def train(self, start_epoch, end_epoch, print_test_accuracy = True,
              print_train_accuracy = True, data_augmentation = True):
        
        model = self.model
        sess = self.sess
        is_correct = tf.equal(tf.argmax(model.outputs,axis = 1), tf.argmax(model.labels_one_hot,axis = 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        for epoch in range(start_epoch, end_epoch):
            total_cost = 0
            idxs = list(range(len(self.train_datas)))
            np.random.shuffle(idxs)

            input_datas = self.train_datas
            input_labels = self.train_labels

            range_list = tqdm_notebook([self.batch_size*x for x in range(0,self.total_batch)])

            for i in range_list:
                if data_augmentation:
                    model_input = sess.run(self.data_aug_imgs, feed_dict = 
                                           {self.image_input_datas:input_datas[idxs[i:i+self.batch_size]]})
                else:
                    model_input = input_datas[idxs[i:i+self.batch_size]]


                _, cost_val = sess.run([self.optimizer,self.loss], feed_dict = {model.input: model_input,
                                                                                model.labels: input_labels[idxs[i:i+self.batch_size]],
                                                                               model.phase_train : True})

                total_cost += cost_val/self.total_batch

            print('Epoch: {}'.format(epoch+1), 'Avg_cost: {}'.format(total_cost))

            if print_test_accuracy:

                
                total_test_batch = int(len(self.test_datas)/self.batch_size)
                acc_val = 0
                range_list = [self.batch_size*x for x in range(0,total_test_batch)]
                for i in range_list:
                    acc_val += sess.run(accuracy, feed_dict = {model.input: self.test_datas[i:i+self.batch_size],
                                                              model.labels: self.test_labels[i:i+self.batch_size],
                                                              model.phase_train : False})
                acc_val/=total_test_batch

                print('Test Accuracy: ', acc_val )
                self.test_acc_list.append(acc_val)
            
            if print_train_accuracy:

                acc_idxs = list(range(len(self.train_datas)))
                random.shuffle(acc_idxs)
                train_acc_datas =self.train_datas[acc_idxs[:10000]]
                train_acc_labels = self.train_labels[acc_idxs[:10000]]
                
                total_train_batch = int(10000/self.batch_size)
                acc_val = 0
                range_list = [self.batch_size*x for x in range(0,total_train_batch)]
                for i in range_list:
                    acc_val+=sess.run(accuracy,feed_dict = {model.input: train_acc_datas[i:i+self.batch_size],
                                                           model.labels: train_acc_labels[i:i+self.batch_size],
                                                           model.phase_train: False})
                acc_val/=total_train_batch
                
                print('Train Accuracy: ', acc_val)
                self.train_acc_list.append(acc_val)
    #         time.sleep(10)
        print('{} epoch done'.format(end_epoch))

        
def show_accuracy(num_epochs, test_acc_list, train_acc_list = None):
    
    epoch_val = np.arange(num_epochs)+1
    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(1,1,1)
    ax.plot(epoch_val,test_acc_list, marker = 'o', label = 'test')
    if train_acc_list is not None:
        ax.plot(epoch_val,train_acc_list, marker = 'o', label = 'train')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    plt.legend(loc = 'best')
    plt.show()

