#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:44:56 2017

@author: ali
"""
import numpy as np


gamma = .9
r=(-0.04)*gamma**5

for i in range(6,6):
    r += gamma**(i)*(-0.04)
    print("i  and  reward:", i, r)
U = r + gamma**6

print U    


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:55:56 2017

-- Programming with PyMC  


@author: ali
"""
import pymc  as pm
import numpy as np
from numpy.linalg import norm
import scipy.stats as stats
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pdb
import random
import time
#
#
#
#p = pm.Uniform('p', lower=0, upper=1)
#p = .05

#def sigmoid(x, W, b):
   
#    z = 1/(1 + np.exp(-(np.dot(W,x)+b)))
#    return z

def logsig(x, W, b):
   
    z = 1./(1. + np.exp(-(np.dot(W,x)+b)))
    return z

def hardlim(x, W, b):
        
    z = np.dot(W,x) + b
    n = len(z)
    for i in range(n):
        if z[i] >= 0. :
            z[i] = 1.
        else:
            z[i] = 0
    return z

def pureline(x,w,b):
    z = np.dot(w, x) + b
    return z

#W = np.array([1 , 2])
#b = np.array([0])
#x = [0,-1]

# My first NNet program  GREETING!
# print "Ali! Welcome to the Neural Networks World"    
#
# Three layer NNET for classification

# forward prop 

#w=np.zeros((2,2,11))

#w_1 = np.array([1, –1, 1, –1, 1, –1, 1, –1, –1, 1, 1],[1, –1, –1, 1, –1, 1, –1, 1, –1, 1, 1])
'''
p = np.zeros((2,1,8))
t = np.zeros((2,1,8))


p[:,:,0] = np.array([[2], [0]])
t[:,:,0] = np.array([[0], [1]])

p[:,:,1] = np.array([[-1], [2]])
t[:,:,1] = np.array([[1], [0]])

p[:,:,2] = np.array([[-2], [1]])
t[:,:,2] = np.array([[1], [0]])

p[:,:,3] = np.array([[1], [1]])
t[:,:,3] = np.array([[0], [0]])

p[:,:,4] = np.array([[1], [2]])
t[:,:,4] = np.array([[0], [0]])

p[:,:,5] = np.array([[2], [-1]])
t[:,:,5] = np.array([[0], [1]])

p[:,:,6] = np.array([[-1], [-1]])
t[:,:,6] = np.array([[1], [1]])

p[:,:,7] = np.array([[-2], [-2]])
t[:,:,7] = np.array([[1], [1]])

#
# initial values
w = np.zeros((2,2))

pdb.set_trace()
alpha = .1

w[0,0] = 1. 
w[1,1]= .9

ind = np.zeros((8,1))

b =np.array([[0.01],[1]])

#cont=0
epsilon = 0.0001
# using Gradient decent considering F = e^2
et = np.zeros((2,8))
et = np.array(et)
a = np.zeros((2,8))
et = np.array(a)
for i in range(8):
#      a  = sigmoid(p[:,:,i],w,b) 
   a[:,i] = (np.dot(w,p[:,:,i]) + b).T
   et[:,i] = (t[:,:,i].T - a[:,i])
     
cont =0
Merr = np.dot(et,et.T)
err = np.sqrt(Merr.trace())
while (err > epsilon):
     print("iteration  :", cont)
#         print("gradient  :", grad)
     print("error  :", np.sqrt(np.dot(et,et.T).trace()))
#         continue
     progress = err - np.sqrt(np.dot(et, et.T).trace())
     err = np.sqrt(np.dot(et,et.T).trace())
     print("progress  :", progress)
     if progress < 0:
           alpha /= 2
     else:
           alpha *=1.2
        
#     else:
#        a = hardlim(p[:,:,i],w,b)
#        e = t[:,:,i] - a
#         aprim = np.multiply(a,1-a)
     ep = np.dot(et[:,0].reshape(2,1), p[:,:,0].T)
     eg = et[:,0].reshape(2,1)
     for i in range(1,8):     
         ep += np.dot(et[:,i].reshape(2,1), p[:,:,i].T)
         eg += (t[:,:,i].T - a[:,i]).T
#         grad = 2* np.multiply(aprim,ep)
#         grad = (2 * ep)
     w = w + 2* alpha*ep
#         b = b + 2*alpha*np.multiply(et,aprim)
     b = b + 2* alpha*eg.reshape(2,1)
#
     
     for i in range(8):
#      a  = sigmoid(p[:,:,i],w,b) 
         a[:,i] = (np.dot(w,p[:,:,i]) + b).T
         et[:,i] = (t[:,:,i].T - a[:,i])
     cont +=1
         
#         if (sum(grad*grad) < epsilon):
#              print("Convergence obtained at point", i)
#              break
     if cont > 10000:
       print("maximum iteration reached at point", i) 
       err = 0
print("Converged with iterations :", cont)
     
      
# for double checking
       
#        ind[cont] = i
#        cont +=1 
# double checking:        
#if cont>0:
#    for i in range(cont):
#        a = hardlim(p[:,:,int(ind[i])],w,b)
#        e = t[:,:,int(ind[i])] - a
#        if (sum(e*e) !=0): 
 #           w = w + alpha*np.dot(e,p[:,:,int(ind[i])].T)
 #           b = b + alpha*e
 #       else:
 #           print("it now works for point: ", ind[i])
        
# testing the Algorithm
'''
#
# Problem for two perceptron case; a 1-2-1 Network
#
#for i in range(21):
#    np.random.rand()

def Qfunc(x):
    return 1. + np.sin((np.pi/4.)*x)




N=50
a1 = np.zeros((2,N))
a2 = np.zeros((1,N))
p = np.zeros((1,N))
p[0,0] = 1.
w1 = np.array([[-.27],[-.41]])
w1_s=w1
w2 = np.array([[.09, -0.17]])
w2_s=w2
b1 = np.array([[-.48], [-.13]])
b1_s=b1
b2 = np.array([[.48]])
b2_s=b2




i=0
a1[:,i] = (logsig(p[0,i],w1,b1)).T
a2[0,i] = pureline(a1[:,i].reshape(2,1),w2,b2)
err = tfunc(p[0,0])-a2[0,0]
# forward propagation network

tol = 0.001


for i in range(N):
    if (i>0):
        p[0,i] = (np.random.rand(1) +2.)/4.
    cont =1
    print("----------------------------")
    print("Trying the point: ", p[0,i])
    
    alpha = .1
    w1_tr = w1_s 
    w2_tr = w2_s
    b1_tr = b1_s
    b2_tr = b2_s
    while (1):
#        pdb.set_trace()
# backpropagation
#        time.sleep(1)   
# computing derivatives

        c = -2.*err
        F2 = 1.
        F1 = np.array([[(1.-a1[0,i])*a1[0,i], 0.],[0., (1.-a1[1,i])*a1[1,i]]])
        s2 = c*F2
        s1 = np.dot(F1, w2.T)*s2
    
# using gradient    

        w2_new = w2_tr - alpha*s2
        b2_new = b2_tr - alpha*s2

        w1_new = w1_tr - alpha*s1
        b1_new = b1_tr - alpha*s1

# foreard propagation        
        a1[:,i] = (logsig(p[0,i],w1_new,b1_new)).T
        a2[0,i] = pureline(a1[:,i].reshape(2,1),w2_new,b2_new)
        err_new = np.abs(tfunc(p[0,i])-a2[0,i])
        prgss = err - err_new
        err = err_new
#        print("Progress & alpha:  ", prgss, alpha)
#        print("----------------------------------------------iter:",cont)
#        print("error  --",err)
        
        if (prgss <= 0.):
           alpha /= 2.
           
#        else:
        w1 = w1_new 
        w2 = w2_new
        b1 = b1_new
        b2 = b2_new
        w1_tr = w1_new 
        w2_tr = w2_new
        b1_tr = b1_new
        b2_tr = b2_new     
             
        
        if (err < tol):
            print("convergence achieved with error:  ", err)
            print("and iteration:  ", cont)
          
            break
            
        elif (cont > 500):
            print("Maximum iteration limit reached:  ", err)

            break
         
        cont += 1   
#    if (err<tol):    
    w1_s += (1./float(N))* (w1 - w1_s) 

    w2_s += (1./float(N))* (w2 - w2_s) 

    b1_s += (1./float(N))* (b1 - b1_s) 

    b2_s += (1./float(N))* (b2 - b2_s)
    err=1. 
#print w , p1

#decision(p1)
#decision(p2)

#print "p1 is apple (1)?", decision(p1)

 

#def frwrd(p):
    
#    a_1 = hardline(p,w_1,b_1)
#    a_2 = hardline(a_1,w_2,b_2)
#    a_3 = hardline(a_2,w_3,b_3)
#
#    return a_3
#    for i in range(1,3):
#        a[i] = hardline(a[i-1],w[i],b[i])

# end
print "---------"
print "Thats it!"







