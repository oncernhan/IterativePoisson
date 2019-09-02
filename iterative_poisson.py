# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:20:14 2019

@author: s4419104
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

#Domain
L = 1.0
n = 101
h = L/(n-1)
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
X, Y = np.meshgrid(x, y)

#Source 
f = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        f[i,j] = 2*(x[i]*(x[i] - 1) + y[j]*(y[j] - 1))

#Analytical solution
u_analytical = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        u_analytical[i,j] = x[i]*y[j]*(x[i] - 1)*(y[j] - 1)

#Initial guess
u0 = np.ones((n, n))/20
u0[0,:] = u0[-1,:] = u0[:,0] = u0[:,-1] = 0

#Plot analytical solution and initial guess
fig = plt.figure(figsize = (16, 6), dpi = 50)
ax = fig.add_subplot(121, projection = '3d')
ax.plot_surface(X, Y, u_analytical, rstride = 5, cstride = 5)
plt.title('Analytical solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim3d(bottom = 0, top = 0.07)

ax = fig.add_subplot(122, projection = '3d')
ax.plot_surface(X, Y, u0, rstride = 5, cstride = 5)
plt.title('Initial guess')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim3d(bottom = 0, top = 0.07)
plt.show()

def error(u):
    error = u - u_analytical
    err = (abs(error)).max()
    
    return err


def jacobi(u, f, h, max_err, max_it):
    """
    Jacobi Iteration Method 
    Inputs:
        u: Initial guess
        f: source term
        h: distance
        max_err: maximum error
        iterations: number of iterations
    Outputs:
        u_n: Next guess
        t: time
        it: number of iterations
        conv: vector of errors
    """
    """Setup time"""
    t = time.time()
    #Initialize variables
    u_n = np.copy(u)
    conv = []
    it = 0
    
    while True:
        it += 1
        
        u_n[1:-1, 1:-1] = 0.25*(u_n[2:, 1:-1] + u_n[0:-2, 1:-1] + u_n[1:-1, 2:] + u_n[1:-1, 0:-2] - f[1:-1, 1:-1]*h**2)
        
        #for i in range(1, n-1):
            #for j in range(1, n-1):
                #u_n[i,j] = 0.25*(u_n[i+1, j] + u_n[i-1, j] + u_n[i, j+1] + u_n[i, j-1] - f[i,j] *h*h)
        
        err = error(u_n)
        conv = np.concatenate((conv, [err]))
        
        if err < max_err:
            break
        
        if it > max_it:
            break
        
    t = time.time() - t
    
    print('Computational time = {0:.5f} s'.format(t))
    print('Number of iterations', it)
    print('Maximum error = {0:.4f}'.format(err))
    
    plt.plot(np.arange(1, it + 1), conv)
    plt.xlabel('Iterations')
    plt.ylabel('Maximum error')
    plt.title('Jacobi Iteration Method')
    plt.show()
    
    return u_n, it, conv, t

u_j, it_j, conv_j, t_j = jacobi(u0,f,h,0.01,1000)
    
    

def gauss_seidel(u, f, h, max_err, max_it):
    """
    Gauss-Seidel Iteration Method
    Inputs:
        u: Initial guess
        f: source term
        h: distance
        max_err: maximum error
        iterations: number of iterations
    Outputs:
        u_n: Next guess
        t: time
        it: number of iterations
        conv: vector of errors
    """
    
    #Initialize variables
    t = time.time()
    u_n = np.copy(u)
    conv = []
    it = 0
    
    while True:
        it += 1
        
        for i in range(1, n-1):
            for j in range(1, n-1):
                u_n[i,j] = 0.25*(u_n[i+1, j] + u_n[i-1, j] + u_n[i, j-1] + u_n[i, j+1] - f[i,j]*h*h)
        
        err = error(u_n)
        conv = np.concatenate((conv, [err]))
        
        if err < max_err:
            break
        if it > max_it:
            break
    
    t = time.time() - t
    
    print('Computational time = {0:5f} s'.format(t))
    print('Number of iterations = {0}'.format(it))
    print('Maximum error = {0:4f} '.format(err))
    
    plt.plot(np.arange(1, it + 1), conv)
    plt.xlabel('Number of iterations')
    plt.ylabel('Maximum error')
    plt.title('Gauss-Seidel Iteration Method')
    plt.show()
    
    return u_n, it, conv, t

u_gs, it_gs, conv_gs, t_gs = gauss_seidel(u0, f, h, 0.01, 1000)

def sor(u, f, h, max_err, max_it, w):
    """
    Successive Over-Relaxation Iteration Method
    Inputs:
        u: Initial guess
        f: source term
        h: distance
        max_err: maximum error
        iterations: number of iterations
    Outputs:
        u_n: Next guess
        t: time
        it: number of iterations
        conv: vector of errors
    """
    #Initialize variables
    t = time.time()
    u_n = np.copy(u)
    it = 0
    conv = []
    
    while True:
        it += 1
        
        for i in range(1, n-1):
            for j in range(1, n-1):
                u_n[i,j] = (1 - w)*u_n[i,j] + 0.25*w*(u_n[i+1, j] + u_n[i-1, j] + u_n[i, j+1] + u_n[i, j-1] - f[i,j]*h*h)
        
        err = error(u_n)
        conv = np.concatenate((conv, [err]))
        
        if err < max_err:
            break
        if it > max_it:
            break
        
    
    t = time.time() - t
    #print(conv.size)
    
    print('Computational time = {0:5f} s'.format(t))
    print('Number of iterations = {0}'.format(it))
    print('Maximum error = {0:4f} '.format(err))
    
    
    plt.plot(np.arange(1, it + 1), conv)
    plt.xlabel('Number of iterations')
    plt.ylabel('Maximum error')
    plt.title('Successive Over-Relaxation Method (SOR) Iteration Method (w = {})'.format(w))
    plt.show()
    
    return u_n, it, conv, t

u_sor, it_sor, conv_sor, t_sor = sor(u0, f, h, 0.01, 1000, 1.97)

    
            
    

    
    
        
    
        
