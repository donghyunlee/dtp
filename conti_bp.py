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

    max_epochs, batch_size, n_batches = 100, 100, 500 # = 50000/100
    nX, nH1, nH2, nH3, nH4, nH5, nH6, nH7, nY = 784, 240, 240, 240, 240, 240, 240, 240, 10 # net architecture

    W1 = rand_ortho((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand_ortho((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand_ortho((nH2,nH3), np.sqrt(6./(nH2+nH3)));  B3 = zeros((nH3,))
    W4 = rand_ortho((nH3,nH4), np.sqrt(6./(nH3+nH4)));  B4 = zeros((nH4,))
    W5 = rand_ortho((nH4,nH5), np.sqrt(6./(nH4+nH5)));  B5 = zeros((nH5,))
    W6 = rand_ortho((nH5,nH6), np.sqrt(6./(nH5+nH6)));  B6 = zeros((nH6,))
    W7 = rand_ortho((nH6,nH7), np.sqrt(6./(nH6+nH7)));  B7 = zeros((nH7,))
    W8 = rand_ortho((nH7, nY), np.sqrt(6./(nH7+ nY)));  B8 = zeros((nY, ))    

    F1 = lambda x : tanh( T.dot( x, W1 ) + B1  ) # layer functions
    F2 = lambda x : tanh( T.dot( x, W2 ) + B2  )
    F3 = lambda x : tanh( T.dot( x, W3 ) + B3  )
    F4 = lambda x : tanh( T.dot( x, W4 ) + B4  )
    F5 = lambda x : tanh( T.dot( x, W5 ) + B5  )
    F6 = lambda x : tanh( T.dot( x, W6 ) + B6  )
    F7 = lambda x : tanh( T.dot( x, W7 ) + B7  )
    F8 = lambda x : sfmx( T.dot( x, W8 ) + B8  )

    # X - input design matrix, Y - labels (not one-hot code)
    X, Y = T.fmatrix(), T.fvector() 

    # feedforward
    H1 = F1(X);  H2 = F2(H1); H3 = F3(H2); 
    H4 = F4(H3); H5 = F5(H4); H6 = F6(H5);
    H7 = F7(H6); P  = F8(H7);

    # cost and error
    cost = NLL( P, Y ) 
    err  = error( predict(P), Y )

    # compute gradients
    g_W8, g_B8 = T.grad( cost, [W8, B8] )
    g_W7, g_B7 = T.grad( cost, [W7, B7] )
    g_W6, g_B6 = T.grad( cost, [W6, B6] )
    g_W5, g_B5 = T.grad( cost, [W5, B5] )
    g_W4, g_B4 = T.grad( cost, [W4, B4] )
    g_W3, g_B3 = T.grad( cost, [W3, B3] )
    g_W2, g_B2 = T.grad( cost, [W2, B2] )
    g_W1, g_B1 = T.grad( cost, [W1, B1] )


    ###### training ######
    i = T.lscalar();
    train_fn = theano.function( [i], [cost, err], on_unused_input='ignore',
        givens={ X : train_x[ i*batch_size : (i+1)*batch_size ],  
                 Y : train_y[ i*batch_size : (i+1)*batch_size ] },
        updates = rms_prop( 
            { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, 
              W3 : g_W3, B3 : g_B3, W4 : g_W4, B4 : g_B4, 
              W5 : g_W5, B5 : g_B5, W6 : g_W6, B6 : g_B6, 
              W7 : g_W7, B7 : g_B7, W8 : g_W8, B8 : g_B8 }, 
            __lr)  )

    # evaluation
    eval_valid = theano.function([], [err], givens={ X : valid_x, Y : valid_y }  )

    eval_test = theano.function([], [err], givens={ X : test_x, Y : test_y }  )

    print
    print 'lr = ', __lr
    print 'epoch cost train_err valid_err test_err time(sec)'


    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }
    for e in range(1,max_epochs+1) :
        monitor['train'].append(  np.array([ train_fn(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 10 == 0 :
            monitor['valid'].append( eval_valid() )
            monitor['test'].append(  eval_test() )

            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1][0], monitor['test'][-1][0], time.time() - t


for i in range(10) : exp(0.000804436870084)
