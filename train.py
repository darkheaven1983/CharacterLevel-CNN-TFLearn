# -*- coding: utf-8 -*-

'''
this code is based on the Character level convolutional neural network for text Classification (X. Zhang et al. 2015)
Preprocessing from https://github.com/NVIDIA/DIGITS/tree/master/examples/text-classification
Implementation in TF by Gonzalo Lima
'''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np
import time

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('DBPedia_train.csv', target_column=0, categorical_labels=True, n_classes=14)
dataVal, labelsVal = load_csv('DBPedia_test.csv', target_column=0, categorical_labels=True, n_classes=14)

ALFABETO = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
FEATURE_LONG = 1024


# Preprocessing function
def preprocess(data, length):
    # Crea diccionario de caracteres
    cdict = {}
    for i,c in enumerate(ALFABETO):
    	cdict[c] = i + 2
    for j, nota in enumerate(data):#nota por nota
	sample = np.ones(FEATURE_LONG) # one by default (i.e. 'other' character)
	count = 0
	a = ''.join(nota)
	for char in a:#caracter por caracter de cada nota
		if char in cdict:			
			sample[count] = cdict[char]
		count +=1
		if count >= FEATURE_LONG-1:
			break
	data[j]=sample.tolist()
	if j%1000 ==0:
		print("notas pre-procesadas: ", j, "/", length)
    print("notas pre-procesadas: ", j, "/", length)
    print("convirtiendo notas en arreglo numérico")
    return np.array(data, dtype=np.float32)



# Preprocess Training data
data = preprocess(data)

#Preprocess Validation data
dataVal=preprocess(dataVal)

W= tflearn.initializations.xavier(uniform=True, seed=None, dtype=tf.float32)
# Building convolutional network
network = input_data(shape=[None, 1024], name='input')
network = tflearn.embedding(network, input_dim=71, output_dim=256)
network = conv_1d(network, 256, 7, padding='valid', scope='conv1', activation='relu', weights_init=W)
network = max_pool_1d(network, 3, strides=3, name='Maxpool1D_1')
network = conv_1d(network, 256, 7, padding='valid', scope='conv2', activation='relu', weights_init=W)
network = max_pool_1d(network, 3, strides=3, name='Maxpool1D_2')
network = conv_1d(network, 256, 3, padding='valid', scope='conv3', activation='relu', weights_init=W)
network = conv_1d(network, 256, 3, padding='valid', scope='conv4', activation='relu', weights_init=W)
network = conv_1d(network, 256, 3, padding='valid', scope='conv5', activation='relu', weights_init=W)
network = conv_1d(network, 256, 3, padding='valid', scope='conv6', activation='relu', weights_init=W)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
#network = max_pool_1d(network, 3, strides=3, name='Maxpool1D_Last')
network = tflearn.fully_connected(network, 512, name='Fullyconected_0')
network = dropout(network, 0.5)
network = fully_connected(network, 14, activation='softmax', name='FullyConected_Last')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', name='target')


# Training
model = tflearn.DNN(network)
#model = tflearn.DNN(network,tensorboard_dir='runs', checkpoint_path='Checkpoints', best_checkpoint_path='BestCheckpoint', tensorboard_verbose=2)

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch = 5, shuffle=True, show_metric=True, batch_size=128)
