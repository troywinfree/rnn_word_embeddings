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
"""

import numpy as np
from scipy.optimize import minimize

import theano
from theano import tensor as T


floatX = theano.config.floatX


# the activation function and softmax
__f = T.nnet.sigmoid
__y = T.nnet.softmax



def rnn(i_t,s_tm1,U,W,b) : 
    """ A single neuron in the recursive network
    
        Input
        
        i_t   - one-of-n index encoding
        s_tm1 - output from the previous stage of the rnn
        U     - weight matrix for x_t
        W     - weight matrix for s_tm1
        b     - bias vector
        
        Output
        
        s_t   - output of the t-th stage of the rnn
    """
    
    return __f(U[:,i_t] + W.dot(s_tm1) + b)

    
def compute_network_outputs(i,s0,V,U,W,b) : 
    """ Builds the recursive network using theano's scan iterator. This 
        is a bidirectional rnn where we simply add the outputs of the 
        rnns in both directions. 
        
        Input
        
        i    - array of one-of-n indices representing a sentence
        s0   - initial output of the rnn
        V    - weight matrix for softmax step
        U    - weight matrix for the input to the recursive neuron
        W    - weight matrix for the previous output of the recursive neuron
        b    - bias vector
        
        Output
        
        y    - uncompiled symbolic theano representation of the rnn
    """
    
    fwrd_rslt,_ = theano.scan(rnn,
                              sequences = [i],
                              outputs_info = [s0],
                              non_sequences = [U,W,b])
    bwrd_rslt,_ = theano.scan(rnn,
                              sequences = [i[::-1]],
                              outputs_info = [s0],
                              non_sequences = [U,W,b])
    
    return __y(T.dot(fwrd_rslt + bwrd_rslt, V.T))


def compute_mean_log_lklyhd_outputs(I,J,s0,V,U,W,b) : 
    """ Builds theano symbolic representation of the mean log likelyhood
        loss function for the embedding model. 
        
        Input
    
        I    - domain inputs as 3D tensor
        J    - range outputs as 3D tensor: rows are assumed to be standard 
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
    def scan_f(i,j,s0,V,U,W,b) : 
        
        outs = compute_network_outputs(i,s0,V,U,W,b)
        
        # y is zero except for one one in each row so it's ok to
        # multiply then sum then take the log

        return T.log(theano.scan(lambda j_t,o : o[j_t],sequences = [j,outs])[0]) 
    
    batch_outputs,_ = theano.scan(scan_f,
                                  sequences = [I,J],
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
                 int_dtype = 'int64',
                 float_dtype = floatX
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
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        
        # seed the random generator if requested
        if seed is not None : 
            np.random.seed(seed)
            
        # collect s0
        if s0 is None : 
            self.s0 = theano.shared(np.zeros(k,dtype = self.float_dtype),name='s0')
        else : 
            self.s0 = theano.shared(np.array(s0,dtype = self.float_dtype),name='s0')
        
        # initialize the weights
        self.V = theano.shared(np.array(np.random.rand(n,k),
                                        dtype = self.float_dtype),name='V')
        self.U = theano.shared(np.array(np.random.rand(k,n),
                                        dtype = self.float_dtype),name='U')
        # initializing with full matrix can lead to vanishing gradients
        self.W = theano.shared(np.array(np.diag(np.random.rand(k)),
                                        dtype=self.float_dtype),name='W')
        self.b = theano.shared(np.array(np.random.rand(k),
                                        dtype = self.float_dtype),name='b')  
        
        # shared variables for mean gradient magnitudes
        self.m_dV_mag = theano.shared(np.array(0.,dtype = self.float_dtype),
                                      name='m_dV_mag')
        self.m_dU_mag = theano.shared(np.array(0.,dtype = self.float_dtype),
                                      name='m_dU_mag')
        self.m_dW_mag = theano.shared(np.array(0.,dtype = self.float_dtype),
                                      name='m_dW_mag')
        self.m_db_mag = theano.shared(np.array(0.,dtype = self.float_dtype),
                                      name='m_db_mag')
        
        # shared variables for computing the loss
        self.loss_accum = theano.shared(np.array(0.,dtype = self.float_dtype),
                                        name='loss_accum')
        self.loss_accum_i = theano.shared(np.array(0.,dtype = self.float_dtype),
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
        self.i = T.vector('i',dtype = self.int_dtype)
        
        self.network_outputs = compute_network_outputs(self.i,self.s0,self.V,
                                                       self.U,self.W,self.b)
        
        
        # build mean log likelyhood loss
        
        # the samples are provided as a tensor to support batching of SGD
        self.I = T.matrix('I',dtype = self.int_dtype)
        self.J = T.matrix('J',dtype = self.int_dtype) # for embedding I = J
        
        self.loss_outputs = compute_mean_log_lklyhd_outputs(self.I,self.J,
                                                            self.s0,self.V,
                                                            self.U,self.W,
                                                            self.b)

        # set up the accumulator for computing the loss in batches
        
        n_minibatch = T.cast(self.I.shape[0],self.float_dtype)
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
        self.eta = T.scalar('eta',dtype = self.float_dtype)
        
        # also including a running average of the gradient magnitudes
        
        self.sgd_i = T.scalar('sgd_i',dtype = self.float_dtype)
        
        dV_mag_accum = (self.dV_mag/(self.sgd_i+1.)
                            + self.m_dV_mag*(self.sgd_i/(self.sgd_i+1.)))
        dU_mag_accum = (self.dU_mag/(self.sgd_i+1.) 
                            + self.m_dU_mag*(self.sgd_i/(self.sgd_i+1.)))
        dW_mag_accum = (self.dW_mag/(self.sgd_i+1.) 
                            + self.m_dW_mag*(self.sgd_i/(self.sgd_i+1.)))
        db_mag_accum = (self.db_mag/(self.sgd_i+1.) 
                            + self.m_db_mag*(self.sgd_i/(self.sgd_i+1.)))
        
        # adding here since we are taking a max of the loss - accumulators
        # do not include the latest values
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
            
        self.network = theano.function(inputs = [self.i],
                                       outputs = self.network_outputs)
        
    def compile_loss(self) : 
        """ Compile the loss
        """
        
        if self.loss is not None : 
            return
            
        self.loss = theano.function(inputs = [self.I,self.J],
                                    outputs = self.loss_outputs,
                                    updates = self.loss_updates)
        
    def compile_grad_loss(self) : 
        """ Compile the gradient of the loss
        """
        
        if self.grad_loss is not None : 
            return
            
        self.grad_loss = theano.function(inputs = [self.I,self.J],
                                         outputs = [self.dV,self.dU,
                                                    self.dW,self.db])
        
    def compile_sgd_update(self) : 
        """ Compile SGD update function
        """
        
        if self.sgd_update is not None : 
            return

        self.sgd_update = theano.function(inputs = [self.I,self.J,
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
        theano_f = theano.function(inputs = [self.I,self.J,
                                             self.eta,self.sgd_i],
                                   outputs = self.loss_outputs,
                                   updates = self.sgd_updates)
        
        self.sgd_update_w_loss = theano_f
                
    def batch_optimize(self,_I,_J,tol = 1E-5) : 
        """ Optimize the model using BFGS. This is only for toy problems and 
            serves as a reality check on stochastic gradient descent. 
            
            This will not work with 'L-BFGS-B' if float_dtype is not float64 
            see https://github.com/scipy/scipy/issues/5832
            
            Input
            
            _I       - matrix of domain index samples
            _J       - matrix of range index samples
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
        
        V = T.matrix('V_bo',dtype = self.float_dtype)
        W = T.matrix('W_bo',dtype = self.float_dtype)
        U = T.matrix('U_bo',dtype = self.float_dtype)
        b = T.vector('b_bo',dtype = self.float_dtype)

        I = theano.shared(np.array(_I,dtype = self.int_dtype),'X_bo')
        J = theano.shared(np.array(_J,dtype = self.int_dtype),'X_bo')
        
        loss_outputs = compute_mean_log_lklyhd_outputs(I,J,self.s0,V,U,W,b)

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
        x0 = np.zeros(2*n*k+k**2+k,dtype = self.float_dtype)
        
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
