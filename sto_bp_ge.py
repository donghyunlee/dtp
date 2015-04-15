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

    max_epochs, batch_size, n_batches = 1000, 100, 500 # = 50000/100
    nX, nH1, nH2, nY = 784, 200, 200, 10 # net architecture

    W1 = rand((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY, ))

    # layer definitions - functions of layers
    F1 = lambda x  : sigm( T.dot( x,  W1 ) + B1  ) # encoding from input to layer 1
    F2 = lambda h1 : sigm( T.dot( h1, W2 ) + B2  ) # encoding from layer 1 to layer 2
    F3 = lambda h2 : sfmx( T.dot( h2, W3 ) + B3  ) # encoding from layer 2 to label probs

    X, Y = T.fmatrix(), T.fvector() # X - input design matrix, Y - labels (not one-hot code)

    # feedforward
    cX = gaussian(X,0.3)
    H1a = T.dot(  cX,   W1 ) + B1
    H1  = sigm(H1a)
    H1s = samp(H1)
    H2a = T.dot(  H1s,  W2 ) + B2
    H2  = sigm(H2a)
    H2s = samp(H2)
    H3a = T.dot(  H2s,  W3 ) + B3

    cost = NLL( sfmx(H3a), Y ) # cost and error

    test_P = F3(samp(F2(samp(F1(X))))) # compute label probs for testing
    err  = error( predict(F3(samp(F2(samp(F1(X)))))), Y ) # compute error for testing

    # grad of parameters using straight-forward gradient estimator
    d_H3a = T.grad( cost, H3a )

    g_B3 = d_H3a.sum(axis=0)
    g_W3 = T.dot( H2s.T, d_H3a )

    d_H2a = T.dot( d_H3a, W3.T ) * H2 * ( 1 - H2 )

    g_B2 = d_H2a.sum(axis=0)
    g_W2 = T.dot( H1s.T, d_H2a )

    d_H1a = T.dot( d_H2a, W2.T ) * H1 * ( 1 - H1 )

    g_B1 = d_H1a.sum(axis=0)
    g_W1 = T.dot( X.T, d_H1a )


    ###### training ######

    i = T.lscalar();
    train_fn = theano.function( [i], [cost, err], on_unused_input='ignore',
        givens={ X : train_x[ i*batch_size : (i+1)*batch_size ],  
                 Y : train_y[ i*batch_size : (i+1)*batch_size ] },
        updates = rms_prop( { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, W3 : g_W3, B3 : g_B3 }, 
                __lr)  )

    #eval_valid = theano.function([i], [err], on_unused_input='ignore', givens={ X : valid_x, Y : valid_y }  )
    #eval_test = theano.function([i], [err],  on_unused_input='ignore', givens={ X : test_x,  Y : test_y  }  )

    valid_probs = theano.function([], test_P, on_unused_input='ignore', givens={ X : valid_x, Y : valid_y }  )
    test_probs  = theano.function([], test_P, on_unused_input='ignore', givens={ X : test_x, Y : test_y }  )

    print
    print __lr
    print 'epoch cost train_err valid_err test_err time(sec)'


    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }

    for e in range(1,max_epochs+1) :

        monitor['train'].append(  np.array([ train_fn(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 10 == 0 :
            avg_test_err  = 100*( np.argmax( np.array([ test_probs()  for i in range(100) ]).mean(axis=0), axis=1) != test_y.get_value() ).mean()            
            avg_valid_err = 100*( np.argmax( np.array([ valid_probs() for i in range(100) ]).mean(axis=0), axis=1) != valid_y.get_value() ).mean()
            monitor['valid'].append( avg_valid_err )
            monitor['test'].append( avg_test_err )

            #monitor['valid'].append( np.array([ eval_valid(i) for i in range(2) ]).mean(axis=0)  )
            #monitor['test'].append(  np.array([ eval_test(i) for i in range(2) ]).mean(axis=0)  )
            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1], monitor['test'][-1], time.time() - t

for i in range(10) : exp(0.00458495466364)

#for i in range(100) :
#    __lr = 10 ** ( ( (-1) - (-4) ) * np.random.random_sample() + (-4) ) #
#    exp(__lr)

