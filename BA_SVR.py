# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:45:01 2018

@author: chang
"""



from BA_kernel_function import kernel_f

from BA_gram import kernel_matrix

import numpy as np
import cvxpy as cvx

class svr_mj:
    '''
    loss: ['epsilon-insensitive','huber','laplacian','gaussian','polynomial','piecewise_polynomial'] defualt: epsilon-insensitive
    kernel: [None,'rbf', 'linear','poly']
    coef0: Independent term in kernel function. in 'linear','poly'
    degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    gamma: Kernel coefficient for ‘rbf’, ‘poly’ 
    p: polynomial/'piecewise_polynomial loss function p
    sigma: for loss function
    '''
    def __init__(self,loss = 'epsilon-insensitive',C = 1.0, epsilon = 1.0, 
                 kernel = None, coef0=1.0, degree=3, gamma=0.1, p=3, sigma=0.1):
        self.loss = loss
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.coef0 = coef0
        self.degree= degree
        self.gamma = gamma
        self.sigma = sigma
        self.p = p
        self.idx1 =[]
        self.idx2 =[]
    
    #model fitting    
    def fit(self,X,y):
        self.X = X
        self.y = y
        
        n = self.X.shape[0]#numper of instances
        #variable for dual optimization problem
        
        alpha = cvx.Variable(n)
        alpha_ = cvx.Variable(n)
        one_vec = np.ones(n)
        
        #object function and constraints of all types of loss fuction
        
        if self.loss == 'huber':
            obj = -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
            obj += self.y*(alpha-alpha_) - self.epsilon*one_vec*(alpha+alpha_) - self.sigma/(2*self.C)*one_vec*(cvx.power(alpha, 2) + cvx.power(alpha_, 2))
            self.svr_obj = cvx.Maximize(obj) 
            
            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()
        
            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]

            self.idx1 = idx1
            self.idx2 = idx2
            print(idx1)

            idx = np.append(idx1,idx2)
            print(idx) 
            self.b = 0
            for idx0 in idx :
                self.b += self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)

     
        elif self.loss == 'laplacian': # epsilon = 0 of epsilon-insensitive
            obj = -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
            obj += self.y*(alpha-alpha_)
            self.svr_obj = cvx.Maximize(obj) 
            
            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0, alpha[i] <= self.C, alpha_[i] <= self.C]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]

            self.idx1 = idx1
            self.idx2 = idx2
            print(idx1)

            idx = np.append(idx1,idx2)
            print(idx) 
            self.b = 0
            for idx0 in idx :
                self.b += self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)

        elif self.loss == 'epsilon-insensitive': 
            obj = -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
            obj +=  self.y*(alpha-alpha_) - self.epsilon*one_vec*(alpha+alpha_)
            
            self.svr_obj = cvx.Maximize(obj) 

            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0, alpha[i] <= self.C, alpha_[i] <= self.C]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()

            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel()  <= self.C, np.array(alpha.value).ravel()  > 0))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() <= self.C, np.array(alpha_.value).ravel() > 0))[0]
            
            self.idx1 = idx1
            self.idx2 = idx2
            idx = np.append(idx1,idx2)

            print(idx) 
            self.b = 0
            for idx0 in idx1 :
                self.b += -self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            for idx0 in idx2 :
                self.b +=  self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)

        #Gaussian loss function               
        elif self.loss == 'gaussian': # sigma = 1 of huber
            obj = -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
#            obj +=  self.y*(alpha-alpha_) - self.epsilon*one_vec*(alpha+alpha_) - 1./(2*self.C)*one_vec*(cvx.power(alpha, 2) + cvx.power(alpha_, 2))
            obj +=  self.y*(alpha-alpha_) - 1./(2*self.C)*one_vec*(cvx.power(alpha, 2) + cvx.power(alpha_, 2))
            self.svr_obj = cvx.Maximize(obj) 

            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]

            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()
        
            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]
            
            self.idx1 = idx1
            self.idx2 = idx2
            
            idx = np.append(idx1,idx2)

            self.b = 0
            for idx0 in idx1 :
                self.b += -self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            for idx0 in idx2 :
                self.b += self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)
           



        elif self.loss == 'polynomial':

            obj = -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
            obj +=   self.y*(alpha-alpha_) - (self.p-1)/(self.p*self.C**(self.p-1))* one_vec*(cvx.power(alpha, self.p/(self.p-1)) + cvx.power(alpha_, self.p/(self.p-1)))
            self.svr_obj = cvx.Maximize(obj) 

            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]

            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()
        
            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]
            
            self.idx1 = idx1
            self.idx2 = idx2
            
            idx = np.append(idx1,idx2)

            self.b = 0
            for idx0 in idx1 :
                self.b += -self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            for idx0 in idx2 :
                self.b += self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)
            
            
            
        elif self.loss == 'piecewise_polynomial':
            obj =  -.5 * cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) 
            obj += self.y*(alpha-alpha_) - (self.p-1)*self.sigma/(self.p*self.C**(self.p-1))* one_vec*(cvx.power(alpha, self.p/(self.p-1)) + cvx.power(alpha_, self.p/(self.p-1)))
            self.svr_obj = cvx.Maximize(obj) 

            self.constraints = []
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()
        
            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]
            
            self.idx1 = idx1
            self.idx2 = idx2
            
            idx = np.append(idx1,idx2)

            self.b = 0
            for idx0 in idx1 :
                self.b += -self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            for idx0 in idx2 :
                self.b += self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)
            
            
        else:
            self.constraints = []
            self.svr_obj = cvx.Maximize(-.5*cvx.quad_form(alpha-alpha_, kernel_matrix(self.X,kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma)) - self.epsilon*one_vec*(alpha+alpha_) + self.y*(alpha-alpha_))
            self.constraints += [cvx.sum(one_vec*alpha - one_vec*alpha_) == 0]
            for i in range(n):
                self.constraints += [alpha[i] >= 0, alpha_[i] >=0, alpha[i] <= self.C, alpha_[i] <= self.C]
            svr = cvx.Problem(self.svr_obj,self.constraints)
            svr.solve()
        
            #alpha & alpha_
            self.a = np.array(alpha.value).flatten()
            self.a_ = np.array(alpha_.value).flatten()
        
            #compute b
            idx1 = np.where(np.logical_and(np.array(alpha.value).ravel() < self.C-1E-5, np.array(alpha.value).ravel() > 1E-5))[0]
            idx2 = np.where(np.logical_and(np.array(alpha_.value).ravel() < self.C-1E-5, np.array(alpha_.value).ravel() > 1E-5))[0]
            
            self.idx1 = idx1
            self.idx2 = idx2
            
            idx = np.append(idx1,idx2)

            self.b = 0
            for idx0 in idx1 :
                self.b += -self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            for idx0 in idx2 :
                self.b += self.epsilon + self.y[idx0] - np.sum([(alpha.value[i]-alpha_.value[i])*kernel_f(self.X[idx0], self.X[i], kernel=self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n)])
            self.b /= len(idx)
            
            
     
    def predict(self,new_X):
        self.results = []
        n = new_X.shape[0]
        n2 = self.X.shape[0]

        for j in range(n):
            self.results += [np.sum([(self.a[i] - self.a_[i]) *kernel_f(new_X[j], self.X[i],
                                     kernel = self.kernel,coef0=self.coef0,degree=self.degree,gamma=self.gamma) for i in range(n2)]) + self.b]   
        return np.array(self.results)
    
    
    def get_idx(self):
        return self.idx1, self.idx2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    