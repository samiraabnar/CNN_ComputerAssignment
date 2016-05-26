import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import pool
import sys
sys.path.append('../../')

from CNN_ComputerAssignment.src.ConvolutionLayer import *
from CNN_ComputerAssignment.src.FullyConnectedLayer import *
from CNN_ComputerAssignment.src.LogisticRegression import *


rng = np.random.RandomState(23455)


class ConvolutionalNetwork(object):

    def __init__(self, batch_size,input_shape= (1,96,96),numbers_of_feature_maps=[1,50,20],filter_shape=(5,5), pooling_size=(2, 2),output_size=10, learning_rate=0.1):
        self.input_shape = input_shape
        self.numbers_of_feature_maps = numbers_of_feature_maps
        self.filter_shape = filter_shape
        self.pooling_size = pooling_size
        self.learning_rate = learning_rate

        self.output_size = output_size
        self.batch_size = batch_size
        self.stride = 1
        self.zero_padding = 0
        self.calculate_image_shape_in_each_layer()


    def calculate_image_shape_in_each_layer(self):
        self.image_shape = np.zeros((len(self.numbers_of_feature_maps),2) , dtype=np.int)

        self.image_shape[0] = (self.input_shape[1],self.input_shape[2])
        for i in np.arange(1,len(self.image_shape),1):
            self.image_shape[i][0] = (((self.image_shape[i-1][0] - self.filter_shape[0] + 2*self.zero_padding)/ self.stride) + 1) / self.pooling_size[0]
            self.image_shape[i][1] = (((self.image_shape[i-1][1] - self.filter_shape[0] + 2*self.zero_padding)/ self.stride) + 1) / self.pooling_size[1]

    def test_convolution(self):
        """
        input: a 4D tensor corresponding to a mini-batch of input images. The shape of the tensor is as follows:
        [mini-batch size, number of input feature maps, image height, image width].
        """
        self.input = T.tensor4(name='input')


        #Weights
        W_shape = (self.numbers_of_feature_maps[1],self.numbers_of_feature_maps[0],self.filter_shape[0],self.filter_shape[1])
        w_bound = np.sqrt(self.numbers_of_feature_maps[0]*self.filter_shape[0]*self.filter_shape[1])
        self.W =  theano.shared( np.asarray(np.random.uniform(-1.0/w_bound,1,0/w_bound,W_shape),dtype=self.input.dtype), name = 'W' )

        #Bias

        bias_shape = (self.numbers_of_feature_maps[1],)
        self.bias = theano.shared(np.asarray(
            np.random.uniform(-.5,.5, size=bias_shape),
            dtype=input.dtype), name ='b')

        #Colvolution

        self.convolution = conv.conv2d(self.input,self.W)
        self.max_pooling = pool.max_pool_2d(
            input=self.convolution,
            ds=self.pooling_size,
            ignore_border=True
        )

        output = T.tanh(self.convolution + self.bias.dimshuffle('x', 0, 'x', 'x'))

        f = theano.function([input], output)

    def build_model(self):

        x = T.matrix('x')
        y = T.ivector('y')


        input_layer = x.reshape((self.batch_size, 1, self.image_shape[0][0], self.image_shape[0][1]))

        self.layers = {}
        self.layers[0] = ConvolutionLayer(
                    random_state=rng,
                    input=input_layer,
                    image_shape=(self.batch_size,self.numbers_of_feature_maps[0],self.image_shape[0][0],self.image_shape[0][1]),
                    filter_shape=(self.numbers_of_feature_maps[1],self.numbers_of_feature_maps[0],self.filter_shape[0],self.filter_shape[1]),
                    pooling_size=self.pooling_size

        )

        self.layers[1] = ConvolutionLayer(
                    random_state = rng,
                    input=self.layers[0].output,
                    image_shape = (self.batch_size, self.numbers_of_feature_maps[1],self.image_shape[1][0],self.image_shape[1][1]),
                    filter_shape = (self.numbers_of_feature_maps[2],self.numbers_of_feature_maps[1],self.filter_shape[0],self.filter_shape[1]),
                    pooling_size = self.pooling_size

        )

        self.layers[2] = FullyConnectedLayer(
                    random_state=rng,
                    input=self.layers[1].output.flatten(2),
                    input_dim=self.numbers_of_feature_maps[2] * self.image_shape[2][0] * self.image_shape[2][1],
                    output_dim=500,
                    activation=T.tanh
        )

        self.layers[3] = LogisticRegression(input=self.layers[2].output, input_dim=self.layers[2].output_dim, output_dim=self.output_size)

        cost = self.layers[3].negative_log_likelihood(y)
        error = self.layers[3].errors(y)
        self.predictions = self.layers[3].predictions
        all_params = self.layers[3].params + self.layers[2].params + self.layers[1].params + self.layers[0].params

        grads = T.grad(cost, all_params)

        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(all_params,grads)]

        self.sgd_step = theano.function([x,y], [cost], updates=updates)

        self.test_model = theano.function([x,y], error)
        self.predict_class = theano.function([x], self.predictions)
