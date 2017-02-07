#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:02:07 2016

@author: troywinfree

RNN for word embedding experiements. This is just a bidirectional version of
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf

In particular:
    s_+(t) = f(Uw(t) + Ws_+(t-1) + b)
    s_-(t) = f(Uw(m-t) + Ws_-(t-1) + b)
    y(t) = g(Vs_+(t) + Vs_-(t))
and
    f(z) = 1 / (1 + e^{-z})
    g(z_m) = e^{z_m} / sum e^{z_k}
    
This version multiplies the full one-of-n encoding by the input weight matrix
wich is inefficient
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
           
        s_+(t) = f(Uw(t) + Ws_+(t-1) + b)
        s_-(t) = f(Uw(m-t) + Ws_-(t-1) + b)
        y(t) = g(Vs_+(t) + Vs_-(t))
        
        and
        
        f(z) = 1 / (1 + e^{-z})
        g(z_m) = e^{z_m} / sum e^{z_k}
        
        This is just a bidirectional version of the model that appears in 
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf
    """
    
    def __init__(self,
                 n,   # dimension of inputs
                 k,   # dimension of embedding space
                 seed = None,
                 s0 = None,
                 dtype = floatX
                 ) : 
        """ Model initializer
        
            Inputs
            
            n     - dimension of the inputs at each recursive layer
            k     - dimension of the embedding space
            seed  - optional seed for random initialization of weight matrices
            s0    - optional initial output to start netword recursion
            dtype - float type for computations
        """
        
        self.n = n
        self.k = k
        self.seed = seed
        self.dtype = dtype
        
        # seed the random generator if requested
        if seed is not None : 
            np.random.seed(seed)
            
        # collect s0
        if s0 is None : 
            self.s0 = theano.shared(np.zeros(k,dtype = self.dtype),name='s0')
        else : 
            self.s0 = theano.shared(np.array(s0,dtype = self.dtype),name='s0')
        
        # initialize the weights
        self.V = theano.shared(np.array(np.random.rand(n,k),
                                        dtype = self.dtype),name='V')
        self.U = theano.shared(np.array(np.random.rand(k,n),
                                        dtype = self.dtype),name='U')
        # initializing with full matrix can lead to vanishing gradients
        self.W = theano.shared(np.array(np.diag(np.random.rand(k)),
                                        dtype=self.dtype),name='W')
        self.b = theano.shared(np.array(np.random.rand(k),
                                        dtype = self.dtype),name='b')  
        
        # shared variables for mean gradient magnitudes
        self.m_dV_mag = theano.shared(np.array(0.,dtype = self.dtype),
                                      name='m_dV_mag')
        self.m_dU_mag = theano.shared(np.array(0.,dtype = self.dtype),
                                      name='m_dU_mag')
        self.m_dW_mag = theano.shared(np.array(0.,dtype = self.dtype),
                                      name='m_dW_mag')
        self.m_db_mag = theano.shared(np.array(0.,dtype = self.dtype),
                                      name='m_db_mag')
        
        # shared variables for computing the loss
        self.loss_accum = theano.shared(np.array(0.,dtype = self.dtype),
                                        name='loss_accum')
        self.loss_accum_i = theano.shared(np.array(0.,dtype = self.dtype),
                                          name='loss_accum_i')
        
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
        self.x = T.matrix('x',dtype = self.dtype)
        
        self.network_outputs = compute_network_outputs(self.x,self.s0,self.V,
                                                       self.U,self.W,self.b)
        
        
        # build mean log likelyhood loss
        
        # the samples are provided as a tensor to support batching of SGD
        self.X = T.tensor3('X',dtype = self.dtype)
        self.Y = T.tensor3('Y',dtype = self.dtype) # for embedding Y = X
        
        self.loss_outputs = compute_mean_log_lklyhd_outputs(self.X,self.Y,
                                                            self.s0,self.V,
                                                            self.U,self.W,
                                                            self.b)

        # set up the accumulator for computing the loss in batches
        
        n_minibatch = T.cast(self.X.shape[0],dtype=self.dtype)
        loss_accum_ipnm = self.loss_accum_i + n_minibatch
        
        self.loss_updates = ((self.loss_accum,
                              (self.loss_outputs*n_minibatch/loss_accum_ipnm
                               + (self.loss_accum 
                                 * self.loss_accum_i/loss_accum_ipnm))),
                             (self.loss_accum_i,loss_accum_ipnm))
        
        # get the gradient of the loss
        
        (self.dV,
         self.dU,
         self.dW,
         self.db) = theano.grad(self.loss_outputs,
                                [self.V,self.U,self.W,self.b])
        
        # get the gradient magnitudes
        
        self.dV_mag = T.sqrt(T.sum(self.dV*self.dV))
        self.dU_mag = T.sqrt(T.sum(self.dU*self.dU))
        self.dW_mag = T.sqrt(T.sum(self.dW*self.dW))
        self.db_mag = T.sqrt(T.sum(self.db*self.db))
        
        # get the sgd update function
        
        # this is the learning parameter
        self.eta = T.scalar('eta',dtype = self.dtype)
        
        # also including a running average of the gradient magnitudes
        
        self.sgd_i = T.scalar('sgd_i',dtype = self.dtype)
        
        dV_mag_accum = (self.dV_mag/(self.sgd_i+1.)
                            + self.m_dV_mag*(self.sgd_i/(self.sgd_i+1.)))
        dU_mag_accum = (self.dU_mag/(self.sgd_i+1.) 
                            + self.m_dU_mag*(self.sgd_i/(self.sgd_i+1.)))
        dW_mag_accum = (self.dW_mag/(self.sgd_i+1.) 
                            + self.m_dW_mag*(self.sgd_i/(self.sgd_i+1.)))
        db_mag_accum = (self.db_mag/(self.sgd_i+1.) 
                            + self.m_db_mag*(self.sgd_i/(self.sgd_i+1.)))
        
        # adding here since we are taking a max of the loss
        self.sgd_updates = ((self.V,self.V + self.eta*self.dV),
                            (self.U,self.U + self.eta*self.dU),
                            (self.W,self.W + self.eta*self.dW),
                            (self.b,self.b + self.eta*self.db),
                            (self.m_dV_mag,dV_mag_accum),
                            (self.m_dU_mag,dU_mag_accum),
                            (self.m_dW_mag,dW_mag_accum),
                            (self.m_db_mag,db_mag_accum))

        # pointers for the compiled functions
        self.network = None
        self.loss = None
        self.grad_loss = None
        self.sgd_update = None
        self.sgd_update_w_loss = None
  
    def compile_network(self) : 
        """ Compile the network  
        """
        if self.network is not None : 
            return
            
        self.network = theano.function(inputs = [self.x],
                                       outputs = self.network_outputs)
        
    def compile_loss(self) : 
        """ Compile the loss
        """
        
        if self.loss is not None : 
            return
            
        self.loss = theano.function(inputs = [self.X,self.Y],
                                    outputs = self.loss_outputs,
                                    updates = self.loss_updates)
        
    def compile_grad_loss(self) : 
        """ Compile the gradient of the loss
        """
        
        if self.grad_loss is not None : 
            return
            
        self.grad_loss = theano.function(inputs = [self.X,self.Y],
                                         outputs = [self.dV,self.dU,
                                                    self.dW,self.db])
        
    def compile_sgd_update(self) : 
        """ Compile SGD update function
        """
        
        if self.sgd_update is not None : 
            return

        self.sgd_update = theano.function(inputs = [self.X,self.Y,
                                                    self.eta,self.sgd_i],
                                          outputs = [],
                                          updates = self.sgd_updates)
        
    def compile_sgd_update_w_loss_output(self) : 
        """ Compile SGD update function with PREVIOUS loss output
        """
        
        if self.sgd_update_w_loss is not None : 
            return

        # note that this returns the PREVIOUS loss value - using 
        # the weights before the update
        theano_f = theano.function(inputs = [self.X,self.Y,
                                             self.eta,self.sgd_i],
                                   outputs = self.loss_outputs,
                                   updates = self.sgd_updates)
        
        self.sgd_update_w_loss = theano_f
                
    def batch_optimize(self,_X,_Y,tol = 1E-5) : 
        """ Optimize the model using BFGS. This is only for toy problems and 
            serves as a reality check on stochastic gradient descent. 
            
            This will not work with 'L-BFGS-B' if float_dtype is not float64 
            see https://github.com/scipy/scipy/issues/5832
            
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
        
        V = T.matrix('V_bo',dtype = self.dtype)
        W = T.matrix('W_bo',dtype = self.dtype)
        U = T.matrix('U_bo',dtype = self.dtype)
        b = T.vector('b_bo',dtype = self.dtype)

        X = theano.shared(np.array(_X,dtype = self.dtype),'X_bo')
        Y = theano.shared(np.array(_Y,dtype = self.dtype),'X_bo')
        
        loss_outputs = compute_mean_log_lklyhd_outputs(X,Y,self.s0,V,U,W,b)

        loss = theano.function(inputs = [V,U,W,b],
                               outputs = loss_outputs,
                               allow_input_downcast = True)
        
        (dV,dU,
         dW,db) = theano.grad(loss_outputs,[V,U,W,b])
        
        grad_loss = theano.function(inputs = [V,U,W,b],
                                    outputs = [dV,dU,dW,db],
                                    allow_input_downcast = True)
        
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
        x0 = np.zeros(2*n*k+k**2+k,dtype = self.dtype)
        
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
