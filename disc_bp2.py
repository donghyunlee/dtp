import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *

# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = \
    np.load('mnist.pkl')

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(train_y), sharedX(valid_x), \
    sharedX(valid_y), sharedX(test_x),  sharedX(test_y)


def exp(__lr) :

    max_epochs, batch_size, n_batches = 500, 100, 500
    nX, nH1, nH2, nY = 784, 500, 500, 10 # net architecture

    W1 = rand_ortho((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand_ortho((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand_ortho((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY,))

    # layer definitions - functions of layers
    F1 = lambda x  : tanh(  T.dot( x,  W1 ) + B1  ) # encoding from input to layer 1
    F2 = lambda h1 : tanh(  T.dot( h1, W2 ) + B2  ) # encoding from layer 1 to layer 2
    F3 = lambda h2 : sfmx(  T.dot( h2, W3 ) + B3  ) # encoding from layer 2 to label probs


    i = T.lscalar() # minibatch index
    e = T.fscalar() # epochs

    # tensor variables : X - input design matrix, Y - labels (not one-hot code)
    X, Y = T.fmatrix(), T.fvector()

    # feed-forward computing
    H1 = F1(X); H2 = F2(H1); P = F3(H2)

    # cost and error
    cost = NLL( P, Y ) # cost for continuous net

    cost_disc = NLL( F3(F2(sign(F1(X)))), Y ) # cost for discrete net
    err  = error( predict( F3(F2(sign(F1(X)))) ), Y ) # err for discrete net

    # gradients
    g_W3, g_B3 = T.grad( cost, [W3, B3] )
    g_W2, g_B2 = T.grad( cost, [W2, B2] )
    g_W1, g_B1 = T.grad( cost, [W1, B1] )


    ###### training ######

    givens_train = { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                     Y : train_y[ i*batch_size : (i+1)*batch_size ] }

    train_ff = theano.function( [i], [cost, cost_disc, err], on_unused_input='ignore',
        givens = givens_train, updates=rms_prop( 
            { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, W3 : g_W3, B3 : g_B3 }, 
        __lr))

    # evaluation
    eval_valid = theano.function([], [err],  on_unused_input='ignore',
        givens={ X : valid_x, Y : valid_y }  )

    eval_test = theano.function([], [err], on_unused_input='ignore',
        givens={ X : test_x,  Y : test_y  }  )

    print
    print 'lr = ', __lr
    print 'epoch cost train_err valid_err test_err time(sec)'


    # training loop
    t = time.time()
    monitor = { 'train' : [], 'train_desc' : 'cost, err', 
                'valid' : [], 'valid_desc' : 'err',
                'test'  : [], 'test_desc'  : 'err', }

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }

    for e in range(1,max_epochs+1) :
        monitor['train'].append(  np.array([ train_ff(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 10 == 0 :
            monitor['valid'].append( eval_valid() )
            monitor['test' ].append( eval_test()  )
            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['train'][-1][2], monitor['valid'][-1][0], monitor['test'][-1][0], time.time() - t


for i in range(10) : exp(0.00324964515659)

#for i in range(100) :
#    __lr = 10 ** ( ( -2 - (-4) ) * np.random.random_sample() + (-4) )
#    exp( __lr)

