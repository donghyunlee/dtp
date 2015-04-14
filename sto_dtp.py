import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *


# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load('/home/dhlee/ML/dataset/mnist/mnist.pkl')

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(train_y), sharedX(valid_x), sharedX(valid_y), sharedX(test_x),  sharedX(test_y)


def exp(__lr0, __lr_f, __lr_b) :

    max_epochs, batch_size, n_batches = 1000, 100, 500 # = 50000/100
    nX, nH1, nH2, nY = 784, 200, 200, 10 # model_dtp_noisy4 - 1500 1500

    W1 = rand((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY, ))
    V2 = rand((nH2,nH1), np.sqrt(6./(nH2+nH1)));  C2 = zeros((nH1,))

    # layer definitions - functions of layers
    F1 = lambda x  : sigm( T.dot( x,  W1 ) + B1  ) # encoding from input to layer 1
    F2 = lambda h1 : sigm( T.dot( h1, W2 ) + B2  ) # encoding from layer 1 to layer 2
    F3 = lambda h2 : sfmx( T.dot( h2, W3 ) + B3  ) # encoding from layer 2 to label probs
    G2 = lambda h2 : tanh( T.dot( h2, V2 ) + C2 )# decoding from layer 2 to layer 1


    i, e = T.lscalar(), T.fscalar();

    X, Y = T.fmatrix(), T.fvector() # X - input design matrix, Y - labels (not one-hot code)
    H1 = F1(gaussian(X,0.3)); 
    H2 = F2(samp(H1));
    H2b = samp(H2);
    P = F3(H2b);
    
    cost = NLL_category( P, Y ) # cost and error
    test_P = F3(samp(F2(samp(F1(X)))))
    err  = error( predict(test_P), Y )
    #err  = error( predict(F3(samp(F2(samp(F1(X)))))), Y )

    H2b_ = H2b - __lr0*T.grad( cost, H2b ) # making the first target
    H2_ = H2 - H2b + H2b_
    H1_ = H1 - G2(H2) + G2(H2_)

    #H1_ = H1 - G2(H2b) + G2(H2b_)

    # grad of parameters using layer-wise target costs
    g_W3, g_B3 = T.grad( cost,        [W3, B3], consider_constant=[H2] )
    g_W2, g_B2 = T.grad( mse(H2,H2_), [W2, B2], consider_constant=[H2_,H1] )
    g_W1, g_B1 = T.grad( mse(H1,H1_), [W1, B1], consider_constant=[H1_] )


    H1_c = gaussian(H1,0.5/(1.+e/100.)); 
    H2_c = F2(samp(H1_c));

    g_V2, g_C2 = T.grad( mse( G2(H2_c), H1_c ), [V2, C2], consider_constant=[H2_c,H1_c] )

    givens_train = { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                     Y : train_y[ i*batch_size : (i+1)*batch_size ] }

    train_inv = theano.function( [i,e], [], givens = givens_train,
        on_unused_input='ignore', updates=rms_prop( { V2 : g_V2, C2 : g_C2  }, __lr_b ) )

    train_ff_sync = theano.function( [i], [cost, err], givens = givens_train, 
        on_unused_input='ignore', updates=rms_prop( { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, W3 : g_W3, B3 : g_B3 }, __lr_f ))

    valid_probs = theano.function([], test_P, on_unused_input='ignore', givens={ X : valid_x, Y : valid_y }  )
    test_probs  = theano.function([], test_P, on_unused_input='ignore', givens={ X : test_x, Y : test_y }  )

    print
    print __lr0, __lr_f, __lr_b
    print 'epochs cost train_err valid_err test_err time'

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }
    for e in range(1,max_epochs+1) :
        for i in range(n_batches) : train_inv(i,e)
        monitor['train'].append(  np.array([ train_ff_sync(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 100 == 0 :
            avg_test_err  = 100*( np.argmax( np.array([ test_probs()  for i in range(100) ]).mean(axis=0), axis=1) != test_y.get_value() ).mean()            
            avg_valid_err = 100*( np.argmax( np.array([ valid_probs() for i in range(100) ]).mean(axis=0), axis=1) != valid_y.get_value() ).mean()
            monitor['valid'].append( avg_valid_err )
            monitor['test'].append( avg_test_err )

            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1], monitor['test'][-1], time.time() - t



for i in range(10) : exp(58.4385169576, 0.00308740019447, 0.00000271629261947)

#exp(58.4385169576, 0.00308740019447, 0.0)

#exp(0.14385169576, 0.308740019447, 0.0)



