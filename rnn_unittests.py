#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:19:49 2017

@author: troywinfree
"""

import unittest

import numpy as np

from rnn import embedding_model
from rnn_dense_input import embedding_model as embedding_model_dense


class test_rnn(unittest.TestCase) :
    """
    unit tests for rnn
    """
    
    @classmethod
    def setUpClass(cls) : 
        """
        initialize the embedding model
        """
        
        cls.int_dtype = 'int16'
        cls.float_dtype = 'float64'
        
        cls.n = 3 # dimension of inputs
        cls.m = 4 # number of recursive layers in the model
        cls.k = 2 # dimension of embedding space
        cls.batch_n = 500 # batch size
        cls.eta = np.array(0.1,dtype=cls.float_dtype) # learning hyperparameter (scale on the gradient)
        cls.seed = 102 # seed for random initialization of the weight matrices
        
        # initialize the model
        cls.model = embedding_model(cls.n,cls.k,cls.seed,
                                    int_dtype = cls.int_dtype,
                                    float_dtype = cls.float_dtype)
        
        cls.model_dense = embedding_model_dense(cls.n,cls.k,cls.seed,
                                                dtype = cls.float_dtype)
        
        # save the weight matrices
        cls.V0 = cls.model.V.get_value()
        cls.U0 = cls.model.U.get_value()
        cls.W0 = cls.model.W.get_value()
        cls.b0 = cls.model.b.get_value()
        
        # get some fake data
        cls.X = np.zeros((cls.batch_n,cls.m,cls.n),dtype = cls.float_dtype)
        cls.Y = np.zeros((cls.batch_n,cls.m,cls.n),dtype = cls.float_dtype)
        
        cls.I = np.zeros((cls.batch_n,cls.m),dtype=cls.int_dtype)
        cls.J = np.zeros((cls.batch_n,cls.m),dtype=cls.int_dtype)
        
        for i in range(cls.batch_n) : 
            for j in range(cls.m) : 
                
                cls.I[i,j] = np.random.randint(0,cls.n)
                cls.J[i,j] = np.random.randint(0,cls.n)
                
                cls.X[i,j,cls.I[i,j]] = 1.
                cls.Y[i,j,cls.J[i,j]] = 1.
    
    def test_compare_models(self) : 
        """
        compare the dense model to the index model
        """
        
        EPSILON = 1E-10
        
        self.model.compile_network()
        self.model_dense.compile_network()
        
        for i in range(self.batch_n) : 
            
            d = (self.model.network(self.I[i]) 
                    - self.model_dense.network(self.X[i]))
            self.assertLess(np.fabs(np.max(d)),EPSILON)
        
        self.model.compile_loss()
        self.model_dense.compile_loss()
        
        d = (self.model.loss(self.I,self.J) 
                    - self.model_dense.loss(self.X,self.Y))
        
        self.assertLess(d,EPSILON)
    
    def test_gradient_descent(self) : 
        """
        test that we can get to the BFGS opt value with gradient descent
        in less than 50 steps
        """
    
        tol = 1E-5
        
        # run an optimization using BFGS
        (opt_res, 
         V_opt, 
         U_opt, 
         W_opt, 
         b_opt) = self.model.batch_optimize(self.I,self.J,tol)
        
        # compile the sgd update
        self.model.compile_sgd_update_w_loss_output()

        # now do some gradient descent
        loss = -np.inf
        for i in range(50) : 
            loss = self.model.sgd_update_w_loss(self.I,self.J,
                                                self.eta,float(i))
            if -loss <= opt_res.fun : 
                break
            
        # make sure the loss is better than the BFGS loss
        self.assertLessEqual(-loss,opt_res.fun)
        
        # make sure the weights are close
        
        self.assertLess(np.max(np.fabs(V_opt-self.model.V.get_value())),0.01)
        self.assertLess(np.max(np.fabs(U_opt-self.model.U.get_value())),0.01)
        self.assertLess(np.max(np.fabs(W_opt-self.model.W.get_value())),0.01)
        self.assertLess(np.max(np.fabs(b_opt-self.model.b.get_value())),0.01)
              
    def test_batch_loss_computation(self) : 
        """
        test that the batch loss computation code is working correctly
        """
        
        self.model.compile_loss()
        
        self.model.loss_accum.set_value(0.)
        self.model.loss_accum_i.set_value(0.)
        
        loss = self.model.loss(self.I,self.J)
        
        self.assertEqual(self.model.loss_accum.get_value(),loss)
        self.assertEqual(self.model.loss_accum_i.get_value(),float(len(self.I)))
        
        self.model.loss_accum.set_value(0.)
        self.model.loss_accum_i.set_value(0.)
        
        for i in range(5) : 
            
            i0 = i*100
            i1 = (i+1)*100
            
            self.model.loss(self.I[i0:i1],self.J[i0:i1])
            
        self.assertLess(np.fabs(self.model.loss_accum.get_value()-loss),1E-14)
        self.assertEqual(self.model.loss_accum_i.get_value(),float(len(self.I)))
        
    def test_mean_grad_magnitudes(self) : 
        """
        test that the mean gradient magnitude computations are correct
        """
        
        EPSILON = 1E-10
        
        # compile the sgd update function
        self.model.compile_sgd_update()
        
        # compile the gradient function
        self.model.compile_grad_loss()
        
        # set all the shared variables to zero
        self.model.m_dV_mag.set_value(0.)
        self.model.m_dU_mag.set_value(0.)
        self.model.m_dW_mag.set_value(0.)
        self.model.m_db_mag.set_value(0.)
        
        # do the updates and accumulate means
        
        dV_mean = 0.
        dU_mean = 0.
        dW_mean = 0.
        db_mean = 0.
        
        n_iters = 50
        for i in range(n_iters) : 
            
            (dV_mag,
             dU_mag,
             dW_mag,
             db_mag) = [np.sqrt(np.sum(d*d)) 
                            for d in self.model.grad_loss(self.I,self.J)]
    
            dV_mean += dV_mag
            dU_mean += dU_mag
            dW_mean += dW_mag
            db_mean += db_mag
            
            self.model.sgd_update(self.I,self.J,
                                  self.eta,float(i))

        dV_mean /= float(n_iters)
        dU_mean /= float(n_iters)
        dW_mean /= float(n_iters)
        db_mean /= float(n_iters)
        
        self.assertLess(np.fabs(dV_mean-self.model.m_dV_mag.get_value()),
                                EPSILON)
        self.assertLess(np.fabs(dU_mean-self.model.m_dU_mag.get_value()),
                                EPSILON)
        self.assertLess(np.fabs(dW_mean-self.model.m_dW_mag.get_value()),
                                EPSILON)
        self.assertLess(np.fabs(db_mean-self.model.m_db_mag.get_value()),
                                EPSILON)
        
        
if __name__ == '__main__':
    unittest.main()
    