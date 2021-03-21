# Numpy Multi Perceptron
 This is a quick project trying to achieve a Multi-Layer Perceptron only using Numpy available features and maybe later use the Cupy library to speed up its process to GPU-Tensorflow like training speed. As mentionned it is a Multi-Layer Perceptron that is currently only using the Numpy library as its backbone and is being trained with the MNIST digit database.
 
 This code contains :
 
 - **main.py** : .py file containing the **Layer_UniD** ( 1D Neuron Layer ) and the **MLP** (Entire Model ) classes as well as the training script for a given model.
 
 - **database.py** : .py file containing all the classes and functions handling the creation and extraction of data from the MNIST database.
 
 - **mnist_test.txt** : Text file containing MNIST digit testing data
 
 - **mnist_train.txt** : Text file containing MNIST digit training data
 
 - **model_save.txt** : Text file containg an example of a save format I will try to implement at a later date ( the model cannot be saved currently ).
 
 - **Sans titre.jpg** : Modifiable image that is used to test by yourself the ability of the model to predict random digits.

The model cannot be saved currently but I will try toadd that feature later using the format in **model_save.txt**

# Prerequisites :
This code works using Numpy, Matplotlib, PIL, time and os.
