# PROJECT SEPTEMBRE 2020
# NUMPY MLP AI / MAIN
# By Enguerran VIDAL

# This file contains the main class of this project as well as an example script.

###############################################################
#                           IMPORTS                           #
###############################################################

# DATA MANIPULATION AND PLOTTING MODULES
import numpy as np
import matplotlib.pyplot as plt

# FILE HANDLING MODULES
import os
import sys
import time
from PIL import Image

# PYTHON FILES
from database import*

###############################################################
#                        LAYER CLASS                          #
###############################################################

class Layer_UniD():
    ''' 1D Layer Object usable by Multi-Layer Perceptron types of AI '''
    def __init__(self,n_neurons,n_inputs,weights=[],biases=[],act_funct='Sigmoid'):
        act_funct=activation_function(act_funct)
        self.activation=act_funct[0]
        self.d_activation=act_funct[1]
        # Layer Size
        self.n_neurons=n_neurons
        self.n_inputs=n_inputs
        # Weights and biases matrices
        if type(weights)==type([]):
            self.weights=np.random.randn(n_neurons,n_inputs)
        else:
            assert weights.shape==(n_neurons,n_inputs)
            self.weights=np.float32(weights)
        if type(biases)==type([]):
            self.biases=np.random.randn(n_neurons,1)
        else:
            assert biases.shape==(n_neurons,1)
            self.biases=np.float32(biases)

    def feed(self,x):
        return self.activation(np.matmul(self.weights,x)+self.biases)

###############################################################
#                         MLP CLASS                           #
###############################################################

class MLP():
    ''' Multi-Layer Perceptron AI Model capable of Training through backpropagation algorithms.'''
    def __init__(self,layers=[],load=False,model_file=None):
        self.current_dir=os.path.dirname(os.path.abspath(__file__))
        assert type(layers)==type([])
        self.n_layers=len(layers)
        self.layers=layers
        self.passed_epochs=0
        self.load=load
        self.model_file=model_file
        if load==True:
            assert type(model_file)!=type(None),"No file has been specified to load the Neural Network from."
            self.load_model(model_file)
        else:
            self.trained=False

    def __str__(self):
        ''' NOT OPERATIVE YET , prints out the model's layout '''
        print('0')

    def add_layer(self,new_layer):
        ''' Adds a Layer_UniD() type as a new layer '''
        if len(self.layers)!=0:
            assert new_layer.n_inputs==self.layers[-1].n_neurons
        self.layers.append(new_layer)
        self.n_layers=self.n_layers+1

    def feed_forward(self,x):
        ''' Feeding algorithm'''
        assert x.shape[0]==self.layers[0].n_inputs
        activations=[]#z
        outputs=[x]#a
        for i in range(self.n_layers):
            activations.append(np.matmul(self.layers[i].weights,x)+self.layers[i].biases)
            x=self.layers[i].activation(activations[-1])
            outputs.append(x)
        return activations,outputs       
        
    def backpropagation(self,activations,outputs,batch_y):
        ''' Backpropagation algorithm transmitting the errors on the last layer.'''
        deltas_W=[]
        deltas_B=[]
        deltas=[None]*self.n_layers
        deltas[-1]=(batch_y-outputs[-1])*self.layers[-1].d_activation(activations[-1])
        for i in reversed(range(len(deltas)-1)):
            deltas[i]=np.matmul(self.layers[i+1].weights.T,deltas[i+1])*self.layers[i].d_activation(activations[i])
        batch_size=batch_y.shape[1]
        for i in range(self.n_layers):
            deltas_B.append(np.matmul(deltas[i],np.ones(shape=(batch_size,1)))/float(batch_size))
            deltas_W.append(np.matmul(deltas[i],outputs[i].T)/float(batch_size))
        return deltas_W,deltas_B

    def train(self,database,n_epochs,learning_rate,batch_size=100):
        ''' Trains the AI Model from a provided MNIST type "database" '''
        print("TRAINING STARTING --------------------------------")
        print("Starting epoch = "+str(self.passed_epochs))
        display_step = 1
        for epoch in range(n_epochs):
            avg_cost = 0.
            total_batch = int(database.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = database.train.next_batch(batch_size)
                activations,outputs=self.feed_forward(batch_x)
                deltas_W,deltas_B=self.backpropagation(activations,outputs,batch_y)
                for i in range(self.n_layers):
                    self.layers[i].weights=self.layers[i].weights+learning_rate*deltas_W[i]
                    self.layers[i].biases=self.layers[i].biases+learning_rate*deltas_B[i]
                avg_cost=avg_cost+np.linalg.norm(outputs[-1]-batch_y)/total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        self.passed_epochs=self.passed_epochs+n_epochs

    def ask(self,fichier,label,plot=False):
        ''' Asks the AI for a prediction and verifies if it is correct.'''
        image=image_translator(fichier)
        image=np.array([image])
        image=np.float32(image)
        _,result=self.feed_forward(image.T)
        if plot==True:
            plt.figure()
            image=import_image(fichier)
            plt.subplot(1,2,1)
            plt.imshow(image,cmap=plt.get_cmap("gray"))
            plt.grid(False)
            plt.subplot(1,2,2)
            plt.grid(False)
            plt.xticks(label)
            plt.bar(range(10),result[-1][:,0], color="#777777")
            plt.ylim([0, 1])
            plt.show()
                
                


###############################################################
#                        FUNCTIONS                            #
###############################################################

def activation_function(func):
    ''' Returns an activation function and its derivative as lambdas'''
    if func=='ReLU':
        return [lambda x : np.maximum(0,x), lambda x : np.where(x<=0,0,1)]
    if func=='Sigmoid':
        return [lambda x : 1/(1+np.exp(-x)), lambda x : 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))]
    if func=='TanH':
        return [lambda x : np.tanh(x),lambda x : 1/(np.cosh(x)**2)]
    if func=='SoftMax':
        return [lambda x : np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x))), lambda x : 1]


def import_image(fichier):
    '''Imports and returns an image as a numpy array'''
    img = Image.open(fichier).convert('L')
    WIDTH, HEIGHT = img.size
    data = list(img.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    return data


def image_translator(nom):
    ''' Returns an image as a 1D array of normalized values [0,1] '''
    data=import_image(nom)
    image=[]
    for i in range(28):
        for j in range(28):
            pixel=int(data[i][j])/255
            image.append(pixel)
    return image

################################################################################
#--------- SCRIPT ---------#

# Import MNIST data
mnist = Tensorflow_like_Database('mnist_train.txt','mnist_test.txt')
label=[0,1,2,3,4,5,6,7,8,9]

# Creating the AI Model
Model=MLP()
layer1=Layer_UniD(16,784)
layer2=Layer_UniD(16,16)
layer3=Layer_UniD(10,16)
Model.add_layer(layer1)
Model.add_layer(layer2)
Model.add_layer(layer3)

# Training the AI and asking it a questiona as verification
Model.train(database=mnist,n_epochs=1000,learning_rate=0.1)
Model.ask('Sans titre.jpg',label,plot=True)




