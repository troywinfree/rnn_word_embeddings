#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:14:05 2017

@author: troywinfree
"""

import numpy as np

from collections import Counter

from rnn import embedding_model

import theano

import logging

import time

floatX = theano.config.floatX


def get_logger(name,
               level = logging.INFO,
               handler = logging.StreamHandler(),
               formatter = logging.Formatter()) : 
    """ simple helper function for assembling a logger
    
        Inputs : 
            
        name      - a name for the logging object
        level     - logging level for message reporting
        handler   - handler determining where log messages are sent
        formatter - formating object for message writing
        
        Outputs : 
        
        logging object
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def read_sentences(path,n_vocabulary) : 
    """ Read sentences from the disk and collect the words, word counts
        and vocabulary. 
        
        Sentence data is assumed to start with a '<s>' token and end with 
        a list of '<e>' tokens. The top n_vocabulary most common words are 
        extracted from the sentence data to form the vocabulary. Of these, the
        least common word is replaced with a '<u>' (unknown) token and if
        '<e>' is not present in the vocabulary the second least common word 
        is replaced with '<e>'.
        
        Inputs : 
            
        path         - path to the sentence .npz file
        n_vocabulary - top n_vocabulary most common words will form the vocab
        
        Outputs : 
            
        sentences    - array of sentences read from the disk
        words        - all the words present in the sentences
        counts       - the number of occurances of each word in the data
        vocabulary   - the vocabulary
    """
    
    if n_vocabulary < 2 : 
        raise ValueError('Need more than two words in vocabulary')
    
    # load the sentences - not including start symbol
    sentences = np.load(path)['arr_0'][:,1:]

    # get the words and word counts
    words,counts = np.unique(sentences,return_counts = True)
    
    # get the sort map
    sort_map = np.argsort(counts)
    
    # collect the vocabulary
    
    vocabulary = np.zeros(n_vocabulary,dtype = words.dtype)
    vocabulary = words[sort_map][-n_vocabulary:][::-1]

    if '<e>' not in vocabulary : 
        vocabulary[-2] = '<e>'
        vocabulary[-1] = '<u>'
    else : 
        vocabulary[-1] = '<u>'
        
    return (sentences,words,counts,vocabulary)
        

def get_one_of_n_indices(vocabulary,sentences,
                         dtype = np.int16) : 
    """ Translate the sentence data into index data - each word is replaced
        by its index in the vocabulary array
        
        Inputs : 
            
        vocabulary   - array of vocabulary words
        sentences    - array of sentences
        dtype        - signed integer type for the output array
        
        Outputs : 
            
        index array  - array in which the words in each sentence in the
                       sentences array are replaced by their corresponding
                       index in the vocabulary array
    """
      
    if '<u>' != vocabulary[-1] : 
        raise ValueError("Last member of vocabulary should be '<u>' token")
    
    n_vocab = len(vocabulary)

    if dtype(n_vocab) != n_vocab : 
        raise ValueError("Provided dtype cannot represent size of vocabulary")
    if dtype(-1) != -1 : 
        raise ValueError("Provided dtype must be signed")
    
    # using counter since if a key is not in a counter
    # it returns zero, so unknown tokens will return zero
    vocab_map = Counter(dict((token,i+1) 
                                for i,token in enumerate(vocabulary[:-1])))

    return np.array([[vocab_map[token]-1 for token in sentence]
                                     for sentence in sentences],dtype = dtype)
        

class word_embedder : 
    """ Class for training a word embedding RNN
    """
    
    def __init__(self,
                 vocabulary,
                 training_data,
                 test_data,
                 model,
                 minibatch_size,
                 eval_test_loss_stride,
                 test_loss_batch_size,
                 seed = None,
                 logger = logging
                 ) :
        """ word_embedder initializer
        
            Inputs : 
                
            vocabulary            - words in the vocabulary, including '<e>'
                                    and '<u>' tags (end-of-sentence, and unknown)
            training_data         - training data as one-of-n indices to the
                                    vocabulary array
            test_data             - test data as one-of-n indices to the 
                                    vocabulary array
            model                 - embedding model
            minibatch_size        - number of sentences to use in a minibatch
            eval_test_loss_stride - number of minibatches between loss test
                                    loss evaluations
            test_loss_batch_size  - number of test sentences in a loss batch
            seed                  - seed for random shuffle after each epoch,
            logger                - logging object
        """
        
        self.vocabulary = vocabulary
        self.n_vocabulary = len(vocabulary)
        self.training_data = training_data
        self.training_data_size = len(training_data)
        self.training_indices = np.arange(self.training_data_size)
        self.test_data = test_data
        self.model = model
        self.minibatch_size = minibatch_size
        self.eval_test_loss_stride = eval_test_loss_stride
        self.test_loss_batch_size = test_loss_batch_size
        
        self.n_test_loss_batches = len(self.test_data)
        self.n_test_loss_batches /= self.test_loss_batch_size
        self.n_test_loss_batches = int(np.ceil(self.n_test_loss_batches))
 
        self.seed = 0
        if seed is not None : 
            self.seed = seed
        np.random.seed(self.seed)
        
        self.logger = logger
        
        self.test_loss = [(-1,-np.inf)]
        self.best_test_loss = (-1,-np.inf)
        self.minibatch_i = 0
        
        self.best_V = self.model.V.get_value()
        self.best_U = self.model.U.get_value()
        self.best_W = self.model.W.get_value()
        self.best_b = self.model.b.get_value()
        
        self.m_dV_mag = []
        self.m_dU_mag = []
        self.m_dW_mag = []
        self.m_db_mag = []
    
    @classmethod
    def init_from_sentences_on_disk(cls,
                                    sentences_path,
                                    n_vocabulary,
                                    n_training_data,
                                    embedding_space_dim,
                                    minibatch_size,
                                    eval_test_loss_stride,
                                    test_loss_batch_size,
                                    nof1_dtype = np.int16,
                                    float_dtype = floatX,
                                    shuffle_seed = None,
                                    model_seed = None,
                                    embedder_seed = None,
                                    logger = logging
                                    ) :
        """ word_embedder initialization from sentence data on disk
        
            Inputs : 
            
            sentences_path        - file path to sentences .npz: sentences
                                    should start with '<s>' tag and end with 
                                    one or more '<e>' tags
            n_vocabulary          - vocabulary will be top n_vocabulary most
                                    common words in sentence data: will always
                                    include '<e>' and '<u>' tokens
            n_training_data       - number of sentences in the training data
            embedding_space_dim   - number of rows in input weights of RNN model 
            minibatch_size        - number of sentences in a minibatch
            eval_test_loss_stride - number of minibatches between test loss
                                    evaluations
            test_loss_batch_size  - number of test sentences in loss batch 
            nof1_dtype            - integer type to use in storing one-of-n
                                    index arrays
            float_dtype           - numeric type for use in rnn model weights
            shuffle_seed          - seed for random shuffle of sentence data
            model_seed            - seed for rnn model weight initialization
            embedder_seed         - seed for random shuffle after each epoch
            logger                - logging object
        """
        
        # read the sentences from the disk
        logger.info('reading sentences from %s'%sentences_path)
        (sentences,
         words,
         counts,
         vocabulary) = read_sentences(sentences_path,n_vocabulary)
        logger.info('reading sentences complete')
        
        # get the one of n indices
        logger.info('computing one-of-n indices')
        data = get_one_of_n_indices(vocabulary,sentences,dtype = nof1_dtype)
        
        # shuffle the data
        
        if shuffle_seed is not None : 
            np.random.seed(shuffle_seed)
            
        np.random.shuffle(data)
        
        # partition the training and test data
        training_data = data[:n_training_data]
        test_data = data[n_training_data:]

        # initialize the rnn model
        logger.info('initializing embedding model')
        model = embedding_model(n_vocabulary,
                                embedding_space_dim,
                                model_seed,
                                nof1_dtype,
                                float_dtype)
        
        return cls(vocabulary,
                   training_data,
                   test_data,
                   model,
                   minibatch_size,
                   eval_test_loss_stride,
                   test_loss_batch_size,
                   embedder_seed)      
       
    def accumulate_loss(self) : 
        """ Accumulate the value of the loss function on the test data across
            self.n_test_loss_batches batches
            
            Output : 
                
            value of the loss function on the test data      
        """
        
        
        self.logger.info("accumulating loss")
        
        self.model.loss_accum.set_value(0.)
        self.model.loss_accum_i.set_value(0.)
        
        for j in range(self.n_test_loss_batches) : 
            
            self.logger.info("batch %d of %d"%(j,self.n_test_loss_batches))
            
            j0 = j*self.test_loss_batch_size
            j1 = (j+1)*self.test_loss_batch_size
            
            batch = self.test_data[j0:j1]

            self.model.loss(batch,batch)
         
        self.logger.info("loss accumulation complete")
            
        return self.model.loss_accum.get_value()
    
    def train(self,
              eta,
              n_minibatches,
              test_loss_epsilon,
              grad_mag_epsilon,
              max_seconds
              ) : 
        """ Perform a training session on the model
        
            Input : 
                
            eta               - learning rate for SGD updates
            n_minibatches     - number of minibatches to train over in this session
            test_loss_epsilon - stop training if test loss falls below this
            grad_mag_epsilon  - stop training if any of the mean gradient magnitudes 
                                falls below this
            max_seconds       - stop training after this many seconds
            
            Output : 
                
            stopping_code     - string indicating why training stopped
        
        """
        
        self.logger.info("training")
        
        # the stopping code
        stopping_code = 'Reached max number of minibatches (%d)'%n_minibatches
        
        # compile the model if need be

        self.model.compile_loss()
        self.model.compile_sgd_update()
        
        start = time.perf_counter()
        
        for i in range(n_minibatches) : 
            
            # get the minibatch
            
            i0 = (i*self.minibatch_size)%self.training_data_size
            i1 = ((i+1)*self.minibatch_size)%self.training_data_size
            
            if i1 < i0 :
                # we reached the end of the data
                i1 = self.training_data_size
            
            self.logger.info('minibatch %d of %d, training indices [%d, %d]'%(i+1,n_minibatches,i0,i1))
            
            minibatch = self.training_data[self.training_indices[i0:i1]]
            
            # compute an update
            
            sgd_i = float(self.minibatch_i)
            
            logger.info('sgd update')
            
            self.model.sgd_update(minibatch,minibatch,eta,sgd_i)
            
            # save gradient magnitudes
            
            self.m_dV_mag.append(self.model.m_dV_mag.get_value())
            self.m_dU_mag.append(self.model.m_dU_mag.get_value())
            self.m_dW_mag.append(self.model.m_dW_mag.get_value())
            self.m_db_mag.append(self.model.m_db_mag.get_value())
            
            # check convergence criteria
            
            if i % self.eval_test_loss_stride == 0 : 
                # evaluate the test loss and check for 'convergence'
                
                self.logger.info('evaluating test loss')
                
                # accumulate the loss
                
                prev_test_loss = self.test_loss[-1][1]
                
                self.test_loss.append((self.minibatch_i,
                                       self.accumulate_loss()))
                
                # save the weights and biases if the loss is the best yet
                if self.test_loss[-1][1] >= self.best_test_loss[1] : 
                    
                    self.best_V = self.model.V.get_value()
                    self.best_U = self.model.U.get_value()
                    self.best_W = self.model.W.get_value()
                    self.best_b = self.model.b.get_value()
                    
                    self.best_test_loss = (self.test_loss[-1][0],
                                           self.test_loss[-1][1])
                
                if np.fabs(prev_test_loss - self.test_loss[-1][1]) < test_loss_epsilon : 
                    
                    # record that we are stopping because of training loss
                    # stabilization
                    stopping_code = 'Training loss stabilized'
                    
                    break
            
            if (self.m_dV_mag[-1] < grad_mag_epsilon
                    and self.m_dU_mag[-1] < grad_mag_epsilon
                    and self.m_dW_mag[-1] < grad_mag_epsilon
                    and self.m_db_mag[-1] < grad_mag_epsilon) : 
                # the average gradient magnitudes are small so let's give up

                self.logger.info('gradient magnitude convergence')

                # accumulate the loss to see if we are better than the 
                # current best
                
                if i % self.eval_test_loss_stride != 0 :
                
                    self.test_loss.append((self.minibatch_i,
                                           self.accumulate_loss()))
                
                # save the weights and biases if the loss is the best yet
                if self.test_loss[-1][1] >= self.best_test_loss[1] : 
                    
                    self.best_V = self.model.V.get_value()
                    self.best_U = self.model.U.get_value()
                    self.best_W = self.model.W.get_value()
                    self.best_b = self.model.b.get_value()
                    
                    self.best_test_loss = (self.test_loss[-1][0],
                                           self.test_loss[-1][1])
                
                # record that we stopped because of gradient magnitudes
                stopping_code = 'Gradient magnitudes < %f'%grad_mag_epsilon
                
                break
 
            if i1 == self.training_data_size :
                # shuffle the data for the next epoch
                self.logger.info('epoch complete')
                np.random.shuffle(self.training_indices)
             
            # check if we have been running too long
            elapsed = time.perf_counter() - start
            if elapsed > max_seconds : 
                
                stopping_code = "Exceeded time limit of %fs. Elapsed = %fs"%(max_seconds,elapsed)
                
            self.minibatch_i += 1

        self.logger.info('stopping code = %s'%stopping_code)
        return stopping_code
