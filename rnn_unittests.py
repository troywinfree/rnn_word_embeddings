#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:19:49 2017

@author: troywinfree
"""

import unittest

import numpy as np

from rnn import embedding_model


class test_rnn(unittest.TestCase) :
    """
    unit tests for rnn
    """
    
    def setUp(self) : 
        """
        initialize the embedding model
        """
        self.n = 3 # dimension of inputs
        self.m = 4 # number of recursive layers in the model
        self.k = 2 # dimension of embedding space
        self.batch_n = 500 # batch size
        self.eta = 0.1 # learning hyperparameter (scale on the gradient)
        self.seed = 102 # seed for random initialization of the weight matrices
        
        # initialize the model
        self.model = embedding_model(self.n,self.k,self.seed)
        
        # save the weight matrices
        self.V0 = self.model.V.get_value()
        self.U0 = self.model.U.get_value()
        self.W0 = self.model.W.get_value()
        self.b0 = self.model.b.get_value()
        
        # get some fake data
        self.X = np.zeros((self.batch_n,self.m,self.n))
        self.Y = np.zeros((self.batch_n,self.m,self.n))
        
        for i in range(self.batch_n) : 
            for j in range(self.m) : 
                self.X[i,j,np.random.randint(0,self.n)] = 1.
                self.Y[i,j,np.random.randint(0,self.n)] = 1.
        
        self.tol = 1E-5
        
        # run an optimization using BFGS
        (self.opt_res, 
         self.V_opt, 
         self.U_opt, 
         self.W_opt, 
         self.b_opt) = self.model.batch_optimize(self.X,self.Y,self.tol)
      
    def test_gradient_descent(self) : 
        """
        test that we can get to the BFGS opt value with gradient descent
        in less than 50 steps
        """
    
        # now do some gradient descent
        loss = -np.inf
        for i in range(50) : 
            [loss] = self.model.sgd_update(self.X,self.Y,self.eta)
            if -loss <= self.opt_res.fun : 
                break
        
        # make sure the loss is better than the BFGS loss
        self.assertLessEqual(-loss,self.opt_res.fun)
        
        # make sure the weights are close
        
        self.assertLess(np.max(np.fabs(self.V_opt-self.model.V.get_value())),
                        0.01)
        self.assertLess(np.max(np.fabs(self.U_opt-self.model.U.get_value())),
                        0.01)
        self.assertLess(np.max(np.fabs(self.W_opt-self.model.W.get_value())),
                        0.01)
        self.assertLess(np.max(np.fabs(self.b_opt-self.model.b.get_value())),
                        0.01)

        
if __name__ == '__main__':
    unittest.main()
    