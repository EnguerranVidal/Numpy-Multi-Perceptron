# PROJECT SEPTEMBRE 2020
# NUMPY MLP AI / DATABASE
# By Enguerran VIDAL

# This file contains databases classes that mimic the methods usable
# by TensorFlow MNIST downloadable databases.

###############################################################
#                           IMPORTS                           #
###############################################################

# DATA MANIPULATION AND PLOTTING MODULES
import matplotlib.pyplot as plt
import numpy as np


###############################################################
#                     DATABASES CLASSES                       #
###############################################################

class Tensorflow_like_Database():
    ''' Parent database class'''
    def __init__(self,train_data,test_data):
        self.train=Train_Database(train_data)
        self.test=Test_Database(test_data)

class Train_Database():
    ''' Training database attached afterwards to the parent's class'''
    def __init__(self,train_data):
        images,labels=file_translator(train_data)
        self.images=np.array(images)
        self.labels=np.array(labels)
        print(self.images.shape)
        print(self.labels.shape)
        self.num_examples=self.images.shape[0]
        assert self.images.shape[0]==self.labels.shape[0]
        self.cursor=0

    def next_batch(self,batch_size):
        images=self.images[self.cursor:self.cursor+batch_size,:]
        labels=self.labels[self.cursor:self.cursor+batch_size,:]
        self.cursor=self.cursor+batch_size
        if self.cursor>=self.num_examples-1:
            self.cursor=0
        return images.T,labels.T

class Test_Database():
    ''' Testing database attached afterwards to the parent's class'''
    def __init__(self,test_data):
        images,labels=file_translator(test_data)
        self.images=np.array(images)
        self.labels=np.array(labels)
        print(self.images.shape)
        print(self.labels.shape)
        self.num_examples=self.images.shape[0]
        assert self.images.shape[0]==self.labels.shape[0]


###############################################################
#                        FUNCTIONS                            #
###############################################################

def file_reading(nom):
    ''' Opens and reads the entire content of a .txt file '''
    file=open(nom,'r')
    t=file.readlines()
    file.close()
    return t

def file_translator(nom):
    ''' Translates the content of a .txt file containing the MNIST database.'''
    t=file_reading(nom)
    n_lines=len(t)
    images=[]
    labels=[]
    for i in range(n_lines):
        if i%2==0:
            label=[]
            digit=int(t[i])
            for j in range(10):
                if j==digit:
                    label.append(1)
                else:
                    label.append(0)
            labels.append(label)
        else:
            image=t[i].split()
            for j in range(784):
                image[j]=int(image[j])/255
            images.append(image)
    return images,labels


