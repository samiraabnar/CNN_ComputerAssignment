import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
from theano.tensor.signal import pool


class ConvolutionLayer(object):
    def __init__(self, input,random_state, image_shape, filter_shape, pooling_size):
        self.input = input
        self.pooling = pooling_size
        self.random_state = random_state
        self.input_dim = np.prod(filter_shape[1:])
        self.output_dim = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(self.pooling)
        self.filter_shape = filter_shape
        self.W, self.bias = self.initialize_weights()

        conv_out = conv.conv2d(input=input,filters=self.W,filter_shape=filter_shape,image_shape=image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=pooling_size, ignore_border=True)

        self.output = T.tanh(pooled_out + self.bias.dimshuffle('x',0,'x','x'))
        self.params = [self.W, self.bias]




    def initialize_weights(self):
         w_values = np.asarray(
             self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                       np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                       self.filter_shape)
             , dtype=theano.config.floatX)

         """if self.activation == theano.tensot.nnet.sigmoid:
             w_values *= 4"""

         W = theano.shared(value=w_values, name="W", borrow="True")

         bias_values = np.zeros(self.filter_shape[0], dtype=theano.config.floatX)
         bias = theano.shared(value=bias_values, name="bias", borrow="True")

         return W,bias