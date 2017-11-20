#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:24:14 2017

 Bayesian Neural Networks (with single neuron) 


@author: Ali

"""
#     ------------------------ Importing Labs ----------------------
import pymc as pm
import numpy as np
import pdb
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


#  -------- ---------   Defining functions  ----- ------ -------

def sigmoid(z):
    return 1./(1.+np.exp(-z))
#    
def gradM(w,e,x,y,alpha):
     
    g = -2.*np.dot(np.multiply(np.multiply(y,1.-y),e),x.T)
    gM = alpha * w + g   
    return gM
#
def findM(w,y,t,alpha):
    
    G  = np.dot((t-y).T,t-y)
    Ew = np.trace(np.dot(w.T,w)/2.)
    return G + alpha*Ew
#
#           -- -- --- ---- defining variables --- -- --- 
#
#   ssetting the prevars 
x = np.zeros((2,1,8))
t = np.zeros((2,1,8))

x[:,:,0] = np.array([[2], [0]])
t[:,:,0] = np.array([[0], [1]])

x[:,:,1] = np.array([[-1], [2]])
t[:,:,1] = np.array([[1], [0]])

x[:,:,2] = np.array([[-2], [1]])
t[:,:,2] = np.array([[1], [0]])

x[:,:,3] = np.array([[1], [1]])
t[:,:,3] = np.array([[0], [0]])

x[:,:,4] = np.array([[1], [2]])
t[:,:,4] = np.array([[0], [0]])

x[:,:,5] = np.array([[2], [-1]])
t[:,:,5] = np.array([[0], [1]])

x[:,:,6] = np.array([[-1], [-1]])
t[:,:,6] = np.array([[1], [1]])

x[:,:,7] = np.array([[-2], [-2]])
t[:,:,7] = np.array([[1], [1]])


# initial values
w = np.zeros((2,2))

w[0,0] = .5 
w[1,1]= .5

b = np.array([[0.],[0.]])

alpha = 0.01
epsilon = 0.09
#  ------------------ Computations ----------------
# 
# MC iteration

L = 10000
# -------- Hamilton iteration
Tau = 20

samp_w = np.zeros((2*L,2))


sw1,sw2 = w.shape

# doing job for 8 points
# pdb.set_trace()
## -----------------------------
# -------------Part 1: Sampling from posterior distribution p(w|D) 
# ------------------------------------

for j in range(1):
     j=5
#     pdb.set_trace()
     z = np.dot(w,x[:,:,j]) + b
     y = sigmoid(z)
     e = t[:,:,j] - y
   
     g = gradM(w,e,x[:,:,j],y,alpha)
     M = findM(w,y,t[:,:,j],alpha)
     count = 0
     for i in range(L):
          
          p = np.random.randn(sw1,sw2)
          H = np.trace(np.dot(p.T,p)/2) + M
#
# Hamilton Monte Carlo Iteration
#    
          for tau in range(Tau):
                p -= epsilon*g/2.
                wnew = w + epsilon*p
# new computations -------------------------------------------          
                z = np.dot(wnew, x[:,:,j]) + b
                y = sigmoid(z)
                e = t[:,:,j] - y
# gradient update  ------------------------------------         
                gnew = gradM(wnew,e,x[:,:,j],y,alpha)
                p -= epsilon*gnew/2.
# objective update ------------------------------------    
          Mnew = findM(wnew,y,t[:,:,j],alpha)
          Hnew = np.trace(np.dot(p.T,p)/2.) + Mnew
          dH = Hnew - H
          print dH
          if (dH < 0):
                  accept = 1
          elif (np.random.uniform(0,1,1) < np.exp(-dH)):
               accept = 1
          else:
               accept = 0
        
          if (accept):
               g = gnew
               w = wnew
#               pdb.set_trace()
               samp_w[2*i:2*i+2,:] = wnew
               count += 1.
               M = Mnew
        
          print('the error:    ',np.sqrt(np.dot(e.T,e))) 
    
# ----------------------------
# --------Approximating Integrals By Monte Carlo
# ----------------------------

yb = 0.




for i in range(int(count)):
#    pdb.set_trace()(1/L)*
    z = np.dot(samp_w[2*i:2*i+2,:],x[:,:,5]) +b 
    yb += (1./count)*sigmoid(z)

print("The approximated value", yb)     
    
    
    
    
     
 
