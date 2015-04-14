import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *

# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = np.load('/home/dhlee/ML/dataset/mnist/mnist.pkl')

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(train_y), sharedX(valid_x), sharedX(valid_y), sharedX(test_x),  sharedX(test_y)


def exp(__lr) :

    max_epochs, batch_size, n_batches = 2000, 100, 500 # = 50000/100
    nX, nH1, nH2, nY = 784, 200, 200, 10 # model_dtp_noisy4 - 1500 1500

    W1 = rand((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY, ))

    # layer definitions - functions of layers
    F1 = lambda x  : sigm( T.dot( x,  W1 ) + B1  ) # encoding from input to layer 1
    F2 = lambda h1 : sigm( T.dot( h1, W2 ) + B2  ) # encoding from layer 1 to layer 2
    F3 = lambda h2 : sfmx( T.dot( h2, W3 ) + B3  ) # encoding from layer 2 to label probs

    X, Y = T.fmatrix(), T.fvector() # X - input design matrix, Y - labels (not one-hot code)

    cX = gaussian(X,0.3)
    H1a = T.dot(  cX,   W1 ) + B1
    H1  = sigm(H1a)
    H1s = samp(H1)
    H2a = T.dot(  H1s,  W2 ) + B2
    H2  = sigm(H2a)
    H2s = samp(H2)
    H3a = T.dot(  H2s,  W3 ) + B3

    cost = negative_log_likelihood( sfmx(H3a), Y ) # cost and error

    test_P = F3(samp(F2(samp(F1(X)))))
    err  = error( predict(F3(samp(F2(samp(F1(X)))))), Y )

    # grad of parameters using layer-wise target costs

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
                            { }, __lr, momentum=.9, averaging_coeff=.95, stabilizer=.0001)  )

    eval_valid = theano.function([i], [err],  on_unused_input='ignore',
        givens={    X : valid_x[ i*5000 : (i+1)*5000 ],  
                    Y : valid_y[ i*5000 : (i+1)*5000 ] }  )

    eval_test = theano.function([i], [err],  on_unused_input='ignore',
        givens={    X : test_x[ i*5000 : (i+1)*5000 ],  
                    Y : test_y[ i*5000 : (i+1)*5000 ] }  )

    valid_probs = theano.function([], test_P, on_unused_input='ignore', givens={ X : valid_x, Y : valid_y }  )
    test_probs  = theano.function([], test_P, on_unused_input='ignore', givens={ X : test_x, Y : test_y }  )

    print
    print __lr


    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }

    for e in range(1,max_epochs+1) :

        monitor['train'].append(  np.array([ train_fn(i) for i in range(n_batches) ]).mean(axis=0)  )
        #if e > 1 and monitor['train'][-1][0] > 20.0 : return
        #if ( e == 10 or e == 100 ) and np.array([ eval_valid(i) for i in range(5) ]).mean(axis=0) > 0.3 : return 

        if e % 10 == 0 :
            avg_test_err  = 100*( np.argmax( np.array([ test_probs()  for i in range(100) ]).mean(axis=0), axis=1) != test_y.get_value() ).mean()            
            avg_valid_err = 100*( np.argmax( np.array([ valid_probs() for i in range(100) ]).mean(axis=0), axis=1) != valid_y.get_value() ).mean()
            monitor['valid'].append( avg_valid_err )
            monitor['test'].append( avg_test_err )

            #monitor['valid'].append( np.array([ eval_valid(i) for i in range(2) ]).mean(axis=0)  )
            #monitor['test'].append(  np.array([ eval_test(i) for i in range(2) ]).mean(axis=0)  )


            #np.set_printoptions(precision=5)
            print "[%4d] cost =" % (e), monitor['train'][-1], monitor['valid'][-1], monitor['test'][-1], " %.2f sec" % (time.time() - t)

        # save the model
        if True and e % 100 == 0 or e == max_epochs:
            with file( "model_bp" , 'wb') as f :
                for obj in [ monitor, W1, B1, W2, B2, W3, B3 ] : 
                    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

            if False and best_measure > monitor['valid'][-1][0] : 
                best_measure = monitor['valid'][-1][0]
                with file( "best_model_orig_tp" , 'wb') as f :
                    hyperparams = [__lr ]
                    for obj in [ monitor, hyperparams, W1, B1, W2, B2, W3, B3 ] : 
                        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)



exp(0.00458495466364)

#for i in range(100) :
#    __lr = 10 ** ( ( (-1) - (-4) ) * np.random.random_sample() + (-4) ) #
#    exp(__lr)

"""
for i in range(200) :
    __lr4 = 10 ** ( ( (-1) - (-6) ) * np.random.random_sample() + (-6) ) #
    __lr1 = 10 ** ( ( (-2) - (-5) ) * np.random.random_sample() + (-5) ) #
    __lr0 = 10 ** ( ( ( 3) - (0) ) * np.random.random_sample() + (0) ) #

    __lr3 = 10 ** ( ( ( 0) - (-2) ) * np.random.random_sample() + (-2) )
    __lr2 = 10 ** ( ( ( 1) - (-1) ) * np.random.random_sample() + (-1) )
    __lambda = 10 ** ( ( (-2) - (-5) ) * np.random.random_sample() + (-5) )

    exp( __lr4, __lr3, __lr2, __lr1, __lr0, __lambda)
"""

"""
for i in range(100) :
    __lr4 = 10 ** ( ( 1 - (-7) ) * np.random.random_sample() + (-7) )
    __lr3 = 10 ** ( ( 2 - (-3) ) * np.random.random_sample() + (-3) )
    __lr2 = 10 ** ( ( 1 - (-5) ) * np.random.random_sample() + (-5) )
    __lr1 = 10 ** ( ( 1 - (-5) ) * np.random.random_sample() + (-5) )
    __lr0 = 10 ** ( ( 2 - (-3) ) * np.random.random_sample() + (-3) )
    __lambda = 10 ** ( ( (-2) - (-7) ) * np.random.random_sample() + (-7) )

    exp( __lr4, __lr3, __lr2, __lr1, __lr0, __lambda)
"""
