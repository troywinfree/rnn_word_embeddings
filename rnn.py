#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:02:07 2016

@author: troywinfree

RNN for word embedding experiements. This is just a bidirectional version of
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf

In particular:
    s_+(t) = f(Uw(t) + Ws(t-1) + b)
    s_-(t) = f(UJw(t) + Ws(t+1) + b)
    y(t) = g(Vs_+(t) + Vs_-(t))
where J has ones on the diagonal from the bottom left to the upper right, and
    f(z) = 1 / (1 + e^{-z})
    g(z_m) = e^{z_m} / sum e^{z_k}
"""

import numpy as np
from scipy.optimize import minimize

import theano
from theano import tensor as T


floatX = theano.config.floatX


# the activation function and softmax
__f = T.nnet.sigmoid
__y = T.nnet.softmax


def rnn(x_t,s_tm1,U,W,b) : 
    """ A single neuron in the recursive network
    
        Input
        
        x_t   - input at the t-th stage of the rnn
        s_tm1 - output from the previous stage of the rnn
        U     - weight matrix for x_t
        W     - weight matrix for s_tm1
        b     - bias vector
        
        Output
        
        s_t   - output of the t-th stage of the rnn
    """
    
    return __f(U.dot(x_t) + W.dot(s_tm1) + b)

    
def compute_network_outputs(x,s0,V,U,W,b) : 
    """ Builds the recursive network using theano's scan iterator. This 
        is a bidirectional rnn where we simply add the outputs of the 
        rnns in both directions. 
        
        Input
        
        x    - matrix of inputs, one for each recursive level of the rnn
        s0   - initial output of the rnn
        V    - weight matrix for softmax step
        U    - weight matrix for the input to the recursive neuron
        W    - weight matrix for the previous output of the recursive neuron
        b    - bias vector
        
        Output
        
        y    - uncompiled symbolic theano representation of the rnn
    """
    
    fwrd_rslt,_ = theano.scan(rnn,
                              sequences = [x],
                              outputs_info = [s0],
                              non_sequences = [U,W,b])
    bwrd_rslt,_ = theano.scan(rnn,
                              sequences = [x[::-1]],
                              outputs_info = [s0],
                              non_sequences = [U,W,b])
    
    return __y(T.dot(fwrd_rslt + bwrd_rslt, V.T))


def compute_mean_log_lklyhd_outputs(X,Y,s0,V,U,W,b) : 
    """ Builds theano symbolic representation of the mean log likelyhood
        loss function for the embedding model. 
        
        Input
    
        X    - domain inputs as 3D tensor
        Y    - range outputs as 3D tensor: rows are assumed to be standard 
               basis vectors
        s0   - initial output of the rnn
        V    - weight matrix for softmax step
        U    - weight matrix for the input to the recursive neuron
        W    - weight matrix for the previous output of the recursive neuron
        b    - bias vector
        
        Output
        
        uncompiled symbolic theano representation of mean log likelyhood loss
    """
    
    # the function to scan over - it just collects the log likelyhood of 
    # the positions indicated by y given x and the weight matrices
    def scan_f(x,y,s0,V,U,W,b) : 
        
        outs = compute_network_outputs(x,s0,V,U,W,b)
        
        # y is zero except for one one in each row so it's ok to
        # multiply then sum then take the log
        return T.log(T.sum(outs*y,axis=1)) 
    
    batch_outputs,_ = theano.scan(scan_f,
                                  sequences = [X,Y],
                                  non_sequences = [s0,V,U,W,b])
    
    return T.mean(batch_outputs)
    
        
class embedding_model : 
    """ Bidirectional RNN for experimenting with word embeddings. The model is
        the following
           
        s_+(t) = f(Uw(t) + Ws(t-1) + b)
        s_-(t) = f(UJw(t) + Ws(t+1) + b)
        y(t) = g(Vs_+(t) + Vs_-(t))
        
        where J has ones on the diagonal from the bottom left to the upper 
        right, and
        
        f(z) = 1 / (1 + e^{-z})
        g(z_m) = e^{z_m} / sum e^{z_k}
        
        This is just a bidirectional version of the model that appears in 
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf
    """
    
    def __init__(self,
                 n,   # dimension of inputs
                 k,   # dimension of embedding space
                 seed = None,
                 s0 = None
                 ) : 
        """ Model initializer
        
            Inputs
            
            n    - dimension of the inputs at each recursive layer
            k    - dimension of the embedding space
            seed - optional seed for random initialization of weight matrices
            s0   - optional initial output to start netword recursion
        """
        
        self.n = n
        self.k = k
        self.seed = seed
        
        # seed the random generator if requested
        if seed is not None : 
            np.random.seed(seed)
            
        # collect s0
        if s0 is None : 
            self.s0 = theano.shared(np.zeros(k),name='s0')
        else : 
            self.s0 = theano.shared(np.array(s0),name='s0')
        
        # initialize the weights
        self.V = theano.shared(np.random.rand(n,k),name='V')
        self.U = theano.shared(np.random.rand(k,n),name='U')
        self.W = theano.shared(np.random.rand(k,k),name='W')
        self.b = theano.shared(np.random.rand(k),name='b')  
        
        # compute the network
        self._compute_network_model()
        
    def _compute_network_model(self) : 
        """ Build the network, loss, grad_loss and sgd_update theano functions.
            More work than is strictly nessecary is done here as the only thing
            that is really needed in order to run sgd (stochastic gradient 
            descent) is the sgd_update function. The network, loss and grad_loss
            functions are compiled since this is experimental code.
        """
        
        # build the network
        x = T.matrix('x',dtype = floatX)
        
        self.network_outputs = compute_network_outputs(x,self.s0,self.V,
                                                       self.U,self.W,self.b)
        self.network = theano.function(inputs = [x],
                                       outputs = self.network_outputs)
        
        # build max likelyhood loss
        
        # the samples are provided as a tensor to support batching of SGD
        X = T.tensor3('X',dtype = floatX)
        Y = T.tensor3('Y',dtype = floatX) # for embedding Y = X
        
        self.loss_outputs = compute_mean_log_lklyhd_outputs(X,Y,
                                                            self.s0,self.V,
                                                            self.U,self.W,
                                                            self.b)

        self.loss = theano.function(inputs = [X,Y],
                                    outputs = self.loss_outputs)
        
        # get the gradient of the loss
        
        (self.dV,
         self.dU,
         self.dW,
         self.db) = theano.grad(self.loss_outputs,
                                [self.V,self.U,self.W,self.b])
        
        self.grad_loss = theano.function(inputs = [X,Y],
                                         outputs = [self.dV,self.dU,
                                                    self.dW,self.db])
        
        # get the sgd update function
        
        # this is the learning parameter
        eta = T.scalar('eta',dtype = floatX)
        
        # adding here since we are taking a max of the loss
        sgd_updates = ((self.V,self.V + eta*self.dV),
                       (self.U,self.U + eta*self.dU),
                       (self.W,self.W + eta*self.dW),
                       (self.b,self.b + eta*self.db))
        
        # note that this returns the PREVIOUS loss value - using 
        # the weights before the update
        self.sgd_update = theano.function(inputs = [X,Y,eta],
                                          outputs = [self.loss_outputs],
                                          updates = sgd_updates)
  
    def batch_optimize(self,_X,_Y,tol = 1E-5) : 
        """ Optimize the model using BFGS. This is only for toy problems and 
            serves as a reality check on stochastic gradient descent. 
            
            Input
            
            _X       - 3D tensor of domain samples
            _Y       - 3D tensor of range samples
            tol      - tolerance for the optimizer
            
            Output
            
            opt_res  - optimization result from the scipy's minimize function
            V_opt    - optimal 'V' weights
            U_opt    - optimal 'U' weights
            W_opt    - optimal 'W' weights
            b_opt    - optimal 'b' weights
        """
        
        # set up the theano functions for evaluating the objective and jacobian
        # for the optimization - if I was a bit more serious about this code
        # I would factor this out as it is lengthy, but since it is just 
        # validation code I have left it as-is
        
        V = T.matrix('V_bo',dtype = floatX)
        W = T.matrix('W_bo',dtype = floatX)
        U = T.matrix('U_bo',dtype = floatX)
        b = T.vector('b_bo',dtype = floatX)

        X = theano.shared(_X,'X_bo')
        Y = theano.shared(_Y,'X_bo')
        
        loss_outputs = compute_mean_log_lklyhd_outputs(X,Y,self.s0,V,U,W,b)

        loss = theano.function(inputs = [V,U,W,b],
                               outputs = loss_outputs)
        
        (dV,dU,
         dW,db) = theano.grad(loss_outputs,[V,U,W,b])
        
        grad_loss = theano.function(inputs = [V,U,W,b],
                                    outputs = [dV,dU,dW,db])
        
        # define the objective an jacobian
        
        def objective(x,n,k) : 
            
            V = x[:n*k].reshape(n,k)
            U = x[n*k:2*n*k].reshape(k,n)
            W = x[2*n*k:2*n*k+k**2].reshape(k,k)
            b = x[2*n*k+k**2:]

            return -loss(V,U,W,b)
            
        def jac(x,n,k) : 
            
            V = x[:n*k].reshape(n,k)
            U = x[n*k:2*n*k].reshape(k,n)
            W = x[2*n*k:2*n*k+k**2].reshape(k,k)
            b = x[2*n*k+k**2:]
        
            dV,dU,dW,db = grad_loss(V,U,W,b)
            
            return -np.concatenate((dV.flatten(),
                                    dU.flatten(),
                                    dW.flatten(),
                                    db.flatten()))
            
        # run the optimization
            
        n = self.n
        k = self.k
        x0 = np.zeros(2*n*k+k**2+k)
        
        x0[:n*k] = self.V.get_value().flatten()
        x0[n*k:2*n*k] = self.U.get_value().flatten()
        x0[2*n*k:2*n*k+k**2] = self.W.get_value().flatten()
        x0[2*n*k+k**2:] = self.b.get_value()
        
        opt_res = minimize(objective,
                           x0,
                           (n,k),
                           'L-BFGS-B',
                           jac,
                           tol = tol)
        
        # collect the optimal weights
        
        V_opt = opt_res.x[:n*k].reshape(n,k)
        U_opt = opt_res.x[n*k:2*n*k].reshape(k,n)
        W_opt = opt_res.x[2*n*k:2*n*k+k**2].reshape(k,k)
        b_opt = opt_res.x[2*n*k+k**2:]
        
        return [opt_res, V_opt, U_opt, W_opt, b_opt]
