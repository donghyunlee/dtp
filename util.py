import theano
import theano.tensor as T
import numpy as np
import cPickle, time, os
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

def one_hot(labels, nC=None):
    nC = np.max(labels) + 1 if nC is None else nC
    code = np.zeros( (len(labels), nC), dtype='float32' )
    for i,j in enumerate(labels) : code[i,j] = 1.
    return code

def sharedX(x) : return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
def randn(shape,mean,std) : return sharedX( mean + std * np.random.standard_normal(size=shape) )
def rand(shape, irange) : return sharedX( - irange + 2 * irange * np.random.rand(*shape) )
def zeros(shape) : return sharedX( np.zeros(shape) ) 

def rand_ortho(shape, irange) : 
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return sharedX(  np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V )) )

def sigm(x) : return T.nnet.sigmoid(x)
def relu(x) : return T.switch( x > 0., x, 0. )
def sfmx(x) : return T.nnet.softmax(x)
def tanh(x) : return T.tanh(x)
def sign(x) : return T.switch(x > 0., 1., -1.)
def softplus(x) : return T.nnet.softplus(x)    
def samp(x) : # x in [0,1]
    rand = RNG.uniform(x.shape, ndim=None, dtype=None, nstreams=None)
    return T.cast( rand < x, dtype='floatX');


def NLL(probs, labels) : # labels are not one-hot code 
    return - T.mean( T.log(probs)[T.arange(labels.shape[0]), T.cast(labels,'int32')] )

def predict(probs) : return T.argmax(probs, axis=1) # predict labels from probs
def error(pred_labels,labels) : return 100.*T.mean(T.neq(pred_labels, labels)) # get error (%)

def mse(x,y) : return T.sqr(x-y).sum(axis=1).mean() # mean squared error
def mce(p,t) : return T.nnet.binary_crossentropy( (p+1.001)/2.002, (t+1.001)/2.002 ).sum(axis=1).mean()

RNG = MRG_RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
def gaussian(x, std, rng=RNG) : return x + rng.normal(std=std, size=x.shape, dtype=x.dtype)

def zero_mask(x,p,rng=RNG) :
    assert 0 <= p and p < 1
    return rng.binomial(p=1-p, size=x.shape, dtype=x.dtype) * x

def rms_prop( param_grad_dict, learning_rate, 
                    momentum=.9, averaging_coeff=.95, stabilizer=.001) :
    updates = OrderedDict()
    for param in param_grad_dict.keys() :

        inc = sharedX(param.get_value() * 0.)
        avg_grad = sharedX(np.zeros_like(param.get_value()))
        avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

        new_avg_grad = averaging_coeff * avg_grad \
            + (1 - averaging_coeff) * param_grad_dict[param]
        new_avg_grad_sqr = averaging_coeff * avg_grad_sqr \
            + (1 - averaging_coeff) * param_grad_dict[param]**2

        normalized_grad = param_grad_dict[param] / \
                T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + stabilizer)
        updated_inc = momentum * inc - learning_rate * normalized_grad

        updates[avg_grad] = new_avg_grad
        updates[avg_grad_sqr] = new_avg_grad_sqr
        updates[inc] = updated_inc
        updates[param] = param + updated_inc

    return updates

