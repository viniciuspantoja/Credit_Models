#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:34:26 2017

@author: ViniciusPantoja
"""

#%%

# Here the author chooses one possible set of weights to solve the problem. 
# I'll change a little bit the functions so we can find one correct answer but setting random
# weights until we find the right one:

# We start with the functions that we will use. First, the linear function with bias;
# second, the relu activation function;
# third, we use an indicator function. This is necessary since we have to compare the solution,
# given some weights, and the Y output. 
# The correct way of doing so is to use gradient descent, but we haven't learned that yet.

import numpy as np
size_hidden = 2
X = np.matrix(([0,0],[1,0],[0,1],[1,1]))
Y = np.matrix(([0],[1],[1],[0]))

def linear(inputs,weights,bias):
    result = np.dot(inputs,weights) + bias
    return result

def relu(inputs):
    result = max(0,inputs)
    return result

def indicator(inputs):
    if inputs >0:
        return 1
    else:
        return 0

#%%
# Now, let's go to the main code. Here we have a while loop, that sets random weights
# each time it is called. 
# We set the decision vector to enter the while. This will be updated in the end of the 
# code by the difference between our prediction and the correct output Y.

final = np.ones((len(Y),1))

while sum((final == [[0],[0],[0],[0]])*1) != 4:
    # Setting the random weights
    W = np.random.randn(2,2)
    omega = np.random.randn(2,1)

    c = np.random.randn(1,2)
    b = np.random.randn(1,1)
    
    # This for loop goes makes the computation for each X point, and than compares it to
    # the correct output
    for i in range(0,len(Y)):
        result_one = relu(linear(X,W,c)[i,0]) 
        result_two = relu(linear(X,W,c)[i,1])
        
        result = np.matrix([result_one,result_two])
        
        result = linear(result,omega,b)
        
        final[i] = indicator(result) - Y[i]

print('The weight of the matrix from input until the first hidden layer is %s, the bias is %s, the weigths of the hidden layer to the output is %s and the second bias is %s'%(W,c,omega,b))

#%%
# Just to check our code, if we use the author's answer insteado of the random answer, the 
# code runs perfectly.

final = np.ones((len(linear(X,W,c)),1))


while sum((final == [[0],[0],[0],[0]])*1) != 4:

    W = np.ones((2,2))
    omega = np.matrix(([1],[-2]))
    c = np.matrix(([0 ,-1]))
    b = 0
    for i in range(0,len(linear(X,W,c))):

        result_one = relu(linear(X,W,c)[i,0])
        result_two = relu(linear(X,W,c)[i,1])

        result = np.matrix([result_one,result_two])
        
        result = linear(result,omega,b)
        
        final[i] = indicator(result) - Y[i]
print('The weight of the matrix from input until the first hidden layer is %s, the bias is %s, the weigths of the hidden layer to the output is %s and the second bias is %s'%(W,c,omega,b))


