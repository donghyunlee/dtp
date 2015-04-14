import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *

# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = \
    np.load('/home/dhlee/ML/dataset/mnist/mnist.pkl')

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(train_y), sharedX(valid_x), sharedX(valid_y), sharedX(test_x),  sharedX(test_y)


def exp(__lr0, __lr_f, __lr_b) :

    max_epochs, batch_size, n_batches = 500, 100, 500
    nX, nH1, nH2, nY = 784, 500, 500, 10

    W1 = rand_ortho((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand_ortho((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand_ortho((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY,))

    V2 = rand_ortho((nH2,nH1), np.sqrt(6./(nH2+nH1)));  C2 = zeros((nH1,))

    # layer definitions - functions of layers
    F1 = lambda x  : tanh(  T.dot( x,  W1 ) + B1  ) 
    F2 = lambda h1 : tanh(  T.dot( sign(h1), W2 ) + B2  ) 
    F3 = lambda h2 : sfmx(  T.dot( h2, W3 ) + B3  ) 

    G2 = lambda h2 : tanh(  T.dot( sign(h2), V2 ) + C2  ) 

    i, e = T.lscalar(), T.fscalar()
    X, Y = T.fmatrix(), T.fvector()

    H1 = F1(X); H2 = F2(H1); P = F3(H2)
    cost = NLL_category( P, Y )
    err  = error( predict(P), Y )

    # target propagation
    H2_ = H2 - __lr0 * T.grad( cost, H2 ) 
    H1_ = H1 - G2(H2) + G2(H2_)

    H1_c = gaussian(H1,0.5/(1.+e/50.)); H2_c = F2(H1_c)
    g_V2, g_C2 = T.grad( mse( G2(H2_c), H1_c ), [V2, C2], consider_constant=[H2_c,H1_c] )

    g_W3, g_B3 = T.grad( cost, [W3, B3], consider_constant=[H2] )
    g_W2, g_B2 = T.grad( mse( H2, H2_ ), [W2, B2], consider_constant=[H2_,H1] )
    g_W1, g_B1 = T.grad( mse( H1, H1_ ), [W1, B1], consider_constant=[H1_,X] )

    ###### training ######

    givens_train = { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                     Y : train_y[ i*batch_size : (i+1)*batch_size ] }

    train_inv = theano.function( [i,e], [], givens = givens_train,
        on_unused_input='ignore', updates=rms_prop( { V2 : g_V2, C2 : g_C2  }, __lr_b ) )

    train_ff_sync = theano.function( [i], [cost, err], givens = givens_train, 
        on_unused_input='ignore', updates=rms_prop( { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, W3 : g_W3, B3 : g_B3 }, __lr_f ))

    # make evaluation function for valid error
    eval_valid = theano.function([i], [err],  on_unused_input='ignore',
        givens={    X : valid_x[ i*5000 : (i+1)*5000 ],  
                    Y : valid_y[ i*5000 : (i+1)*5000 ] }  )

    # make evaluation function for test error
    eval_test = theano.function([i], [err], on_unused_input='ignore',
        givens={    X : test_x[ i*5000 : (i+1)*5000 ],  
                    Y : test_y[ i*5000 : (i+1)*5000 ] }  )

    print
    print __lr0, __lr_f, __lr_b


    # training loop
    t = time.time()
    monitor = { 'train' : [], 'train_desc' : 'cost, err', 
                'valid' : [], 'valid_desc' : 'err',
                'test'  : [], 'test_desc'  : 'err', }

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }

    for e in range(1,max_epochs+1) :
        for i in range(n_batches) : train_inv(i,e)
        monitor['train'].append(  np.array([ train_ff_sync(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 50 == 0 :
            monitor['valid'].append( np.array([ eval_valid(i) for i in range(2) ]).mean(axis=0)  )
            monitor['test' ].append( np.array([ eval_test(i)  for i in range(2) ]).mean(axis=0)  )
            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1][0], monitor['test'][-1][0], time.time() - t


for i in range(10) : exp(27.965482616, 0.000660055346131, 0.000432831736994)

#for i in range(20) :
#    __lr0  = 10 ** ( ( 2 - (0) ) * np.random.random_sample() + (0) )
#    __lr_f = 10 ** ( ( (-2) - (-4) ) * np.random.random_sample() + (-4) )
#    __lr_b = 10 ** ( ( (-3) - (-5) ) * np.random.random_sample() + (-5) )
#    exp( __lr0, __lr_f, __lr_b)

