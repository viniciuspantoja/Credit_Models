gil#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:12:54 2017

@author: ViniciusPantoja
"""
#%%
import os
os.chdir('/Users/ViniciusPantoja/Dropbox/GitHub/jorge_amado')

#Ler o arquivo txt provido pesa UFF (http://www.uff.br/cdme/lpp/lpp-html/lpp-t-br-04.txt). 
data = open('memorias_postumas.txt','r', encoding = 'latin1').read()

# Agora vamos determinar todos os caracteres existentes no texto. a função set() gera um unique

chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

# Agora nós construimos o nosso dicionário, com caracteres e índices. Em deep learning nós temos
# que colocar apenas vetores (cujos elementos são números), então estamos dando um número para cada
# caracter
chars_to_ix = { ch:i for i, ch in enumerate(chars)}
ix_to_chars = { i:ch for i, ch in enumerate(chars)}


import numpy as np

vector_for_char_a = np.zeros((vocab_size,1))
vector_for_char_a[chars_to_ix['a']] = 1

#%%

# Hyperparameters

hidden_size = 100
seq_length = 25
learning_rate = 0.1

# model parameters

Wxh = np.random.randn(hidden_size,vocab_size)*0.01
Whh = np.random.randn(hidden_size,hidden_size)*0.01
Why = np.random.randn(vocab_size,hidden_size)*0.01
bh = np.zeros((hidden_size,1))
by = np.zeros((vocab_size,1))

#%%

def lossFun(inputs,targets,hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    
    hs[-1] = np.copy(hprev)
    loss = 0
    
    #forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1] + bh))
        ys[t] = np.dot(Why,hs[t]) + by
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t],0])
    #print(len(ps))
        
    # Backwards prop
    
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) 
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -=1
        
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        
        dh = np.dot(Why.T,dy) + dhnext
        
        dhraw = (1 - hs[t]*hs[t]) * dh
        
        dbh += dhraw
        
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T,dhraw)
        
    
    for dparam in [dWxh,dWhh,dWhy,dby,dbh]:
        np.clip(dparam, -5, 5, out = dparam)
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]



def sample(h, seed_ix,n):
    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []
    
    for t in range(n):
        h = np.tanh(np.dot(Wxh,x) + np.dot(Whh,h) + bh)
        y = np.dot(Why,h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
        
    txt = ''.join(ix_to_chars[ix] for ix in ixes)
    print ('----\n %s \n----'%(txt, ))
hprev = np.zeros((hidden_size,1))
sample(hprev,chars_to_ix['a'],200)


p = 0
inputs = [chars_to_ix[ch] for ch in data[p:p+seq_length]]
print('inputs = ', inputs)
targets = [chars_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
print ("targets", targets)
#%%
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad                                                                                                                
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0 

#%%                                                                                                                       
while n<=1000000:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  # check "How to feed the loss function to see how this part works
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory                                                                                                                                      
    p = 0 # go from start of data                                                                                                                                                             
  inputs = [chars_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [chars_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # forward seq_length characters through the net and fetch gradient                                                                                                                          
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  # sample from the model now and then                                                                                                                                                        
  if n % 1000 == 0:
    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    sample(hprev, inputs[0], 200)

  # perform parameter update with Adagrad                                                                                                                                                     
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update                                                                                                                   

  p += seq_length # move data pointer                                                                                                                                                         
  n += 1 # iteration counter



