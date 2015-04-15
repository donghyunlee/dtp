import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *

# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) \
    = np.load('mnist.pkl')

train_x, train_y, valid_x, valid_y, test_x, test_y = \
	sharedX(train_x), sharedX(train_y), \
	sharedX(valid_x), sharedX(valid_y), \
	sharedX(test_x),  sharedX(test_y)

def exp(__lr0, __lr_f, __lr_b, __c) :

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

    #V1 = rand_ortho((nH1,nX ), np.sqrt(6./(nX +nH1)));  C1 = zeros((nX, )) # init params
    V2 = rand_ortho((nH2,nH1), np.sqrt(6./(nH1+nH2)));  C2 = zeros((nH1,))
    V3 = rand_ortho((nH3,nH2), np.sqrt(6./(nH2+nH3)));  C3 = zeros((nH2,))
    V4 = rand_ortho((nH4,nH3), np.sqrt(6./(nH3+nH4)));  C4 = zeros((nH3,))
    V5 = rand_ortho((nH5,nH4), np.sqrt(6./(nH4+nH5)));  C5 = zeros((nH4,))
    V6 = rand_ortho((nH6,nH5), np.sqrt(6./(nH5+nH6)));  C6 = zeros((nH5,))
    V7 = rand_ortho((nH7,nH6), np.sqrt(6./(nH6+nH7)));  C7 = zeros((nH6,))
    #V8 = rand_ortho((nY, nH7), np.sqrt(6./(nH7+nY )));  C8 = zeros((nH7,))

    F1 = lambda x : tanh( T.dot( x, W1 ) + B1  ) # layer functions - feedforward
    F2 = lambda x : tanh( T.dot( x, W2 ) + B2  )
    F3 = lambda x : tanh( T.dot( x, W3 ) + B3  )
    F4 = lambda x : tanh( T.dot( x, W4 ) + B4  )
    F5 = lambda x : tanh( T.dot( x, W5 ) + B5  )
    F6 = lambda x : tanh( T.dot( x, W6 ) + B6  )
    F7 = lambda x : tanh( T.dot( x, W7 ) + B7  )
    F8 = lambda x : sfmx( T.dot( x, W8 ) + B8  )

    #G1 = lambda x : sigm( T.dot( x, V1 ) + C1  ) # layer functions - feedback (inverse)
    G2 = lambda x : tanh( T.dot( x, V2 ) + C2  )
    G3 = lambda x : tanh( T.dot( x, V3 ) + C3  )
    G4 = lambda x : tanh( T.dot( x, V4 ) + C4  )
    G5 = lambda x : tanh( T.dot( x, V5 ) + C5  )
    G6 = lambda x : tanh( T.dot( x, V6 ) + C6  )
    G7 = lambda x : tanh( T.dot( x, V7 ) + C7  )
    #G8 = lambda x : tanh( T.dot( x, V8 ) + C8  )

    # X - input design matrix, Y - labels (not one-hot code)
    X, Y = T.fmatrix(), T.fvector() 

    # feedforward
    H1 = F1(X);  H2 = F2(H1); H3 = F3(H2); 
    H4 = F4(H3); H5 = F5(H4); H6 = F6(H5);
    H7 = F7(H6); P  = F8(H7);

    cost = NLL( P, Y ) # cost and error
    err  = error( predict(P), Y )

    # compute layer targets 
    H7_ = H7 - __lr0*T.grad( cost, H7 ) # making the first target
    H6_ = H6 - G7(H7) + G7(H7_)    
    H5_ = H5 - G6(H6) + G6(H6_)
    H4_ = H4 - G5(H5) + G5(H5_)
    H3_ = H3 - G4(H4) + G4(H4_)
    H2_ = H2 - G3(H3) + G3(H3_)
    H1_ = H1 - G2(H2) + G2(H2_)

    # corrupted pairs
    H6_c = gaussian(H6,__c); fH6_c = F7(H6_c);
    H5_c = gaussian(H5,__c); fH5_c = F6(H5_c);
    H4_c = gaussian(H4,__c); fH4_c = F5(H4_c);
    H3_c = gaussian(H3,__c); fH3_c = F4(H3_c);
    H2_c = gaussian(H2,__c); fH2_c = F3(H2_c);
    H1_c = gaussian(H1,__c); fH1_c = F2(H1_c);

    # gradients of feedback (inverse) mapping
    g_V7, g_C7 = T.grad( mse( G7(fH6_c), H6_c ), [V7, C7], consider_constant=[fH6_c,H6_c] )
    g_V6, g_C6 = T.grad( mse( G6(fH5_c), H5_c ), [V6, C6], consider_constant=[fH5_c,H5_c] )
    g_V5, g_C5 = T.grad( mse( G5(fH4_c), H4_c ), [V5, C5], consider_constant=[fH4_c,H4_c] )
    g_V4, g_C4 = T.grad( mse( G4(fH3_c), H3_c ), [V4, C4], consider_constant=[fH3_c,H3_c] )
    g_V3, g_C3 = T.grad( mse( G3(fH2_c), H2_c ), [V3, C3], consider_constant=[fH2_c,H2_c] )
    g_V2, g_C2 = T.grad( mse( G2(fH1_c), H1_c ), [V2, C2], consider_constant=[fH1_c,H1_c] )

    # gradients of feedforward
    g_W8, g_B8 = T.grad( cost,             [W8, B8], consider_constant=[H7] )
    g_W7, g_B7 = T.grad( mse( F7(H6),H7_), [W7, B7], consider_constant=[H7_,H6] )
    g_W6, g_B6 = T.grad( mse( F6(H5),H6_), [W6, B6], consider_constant=[H6_,H5] )
    g_W5, g_B5 = T.grad( mse( F5(H4),H5_), [W5, B5], consider_constant=[H5_,H4] )
    g_W4, g_B4 = T.grad( mse( F4(H3),H4_), [W4, B4], consider_constant=[H4_,H3] )
    g_W3, g_B3 = T.grad( mse( F3(H2),H3_), [W3, B3], consider_constant=[H3_,H2] )
    g_W2, g_B2 = T.grad( mse( F2(H1),H2_), [W2, B2], consider_constant=[H2_,H1] )
    g_W1, g_B1 = T.grad( mse( F1(X), H1_), [W1, B1], consider_constant=[H1_] )

    ###### training ######

    i = T.lscalar();

    givens_train = { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                     Y : train_y[ i*batch_size : (i+1)*batch_size ] }
    updates_inv = rms_prop( { V2 : g_V2, C2 : g_C2, V3 : g_V3, C3 : g_C3, 
            V4 : g_V4, C4 : g_C4, V5 : g_V5, C5 : g_C5, V6 : g_V6, C6 : g_C6,
            V7 : g_V7, C7 : g_C7 }, 
        __lr_b )

    # training feedback(inverse) mapping
    train_inv = theano.function( [i], [], on_unused_input='ignore', 
        givens = givens_train, 
        updates = updates_inv )

    # training feedforward mapping
    train_ff_sync = theano.function([i], [cost, err], givens = givens_train, 
        on_unused_input='ignore', updates=rms_prop( \
            { W1 : g_W1, B1 : g_B1, W2 : g_W2, B2 : g_B2, W3 : g_W3, B3 : g_B3,
              W4 : g_W4, B4 : g_B4, W5 : g_W5, B6 : g_B6, W7 : g_W7, B7 : g_B7,
              W8 : g_W8, B8 : g_B8 }, 
        __lr_f ))

    eval_valid = theano.function([], [err], givens={ X : valid_x, Y : valid_y } )
    eval_test  = theano.function([], [err], givens={ X : test_x,  Y : test_y  } )

    print
    print __lr0, __lr_f, __lr_b, __c
    print 'epochs cost train_err valid_err test_err time(sec)'

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }
    for e in range(1,max_epochs+1) :
        for i in range(n_batches) : train_inv(i)
        monitor['train'].append( np.array([ train_ff_sync(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 10 == 0 :
            monitor['valid'].append( eval_valid() )
            monitor['test'].append(  eval_test()  )
            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1][0], monitor['test'][-1][0], time.time() - t


for i in range(10) : exp(0.327736332653, 0.0148893490317, 0.00501149118237, 0.359829566008)
