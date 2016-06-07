import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import os
import sys
import gzip
import pickle
sys.path.append('../../')


from CNN_ComputerAssignment.src.ConvolutionalNetwork import *


class DigitRecognizer(object):
    def __init__(self,datafile):
        self.batch_size = 500
        self.initialize_data(datafile)
        self.model = ConvolutionalNetwork(batch_size=self.batch_size,input_shape=(1,28,28))


    def initialize_data(self,datafile):
        [self.train_x,self.train_y], [self.dev_x,self.dev_y], [self.test_x,self.test_y] = self.load_mnist_data(datafile)

        self.number_of_batches_in_train = self.train_x.get_value(borrow=True).shape[0] // self.batch_size
        self.number_of_batches_in_dev = self.dev_x.get_value(borrow=True).shape[0] // self.batch_size
        self.number_of_batches_in_test = self.test_x.get_value(borrow=True).shape[0] // self.batch_size

    def train_model(self):
        self.model.build_model()
        epochs = 5

        for epoch in range(epochs):
            # For each training example...
            for i in np.random.permutation(int(self.number_of_batches_in_train)):
                self.model.sgd_step(self.train_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.train_y[i*self.batch_size:(i+1)*self.batch_size].eval())

            test_cost = 0.0
            for i in np.arange(int(self.number_of_batches_in_test)):
                test_cost += self.model.test_model(self.test_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.test_y[i*self.batch_size:(i+1)*self.batch_size].eval())
            train_cost = 0.0
            for i in np.arange(int(self.number_of_batches_in_train)):
                train_cost += self.model.test_model(self.train_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.train_y[i*self.batch_size:(i+1)*self.batch_size].eval())

            test_cost = test_cost / self.number_of_batches_in_test
            train_cost = train_cost / self.number_of_batches_in_train
            print("test cost: ")
            print(test_cost)
            print("train cost: ")
            print(train_cost)

    def step_by_step(self):
        self.model.build_model()
        cost = self.model.sgd_step(self.train_x[0:self.batch_size],np.asanyarray(self.train_y,dtype=np.int32)[0:self.batch_size])
        accuracy = self.model.test_model(self.train_x[0:self.batch_size],np.asanyarray(self.train_y,dtype=np.int32)[0:self.batch_size] )
        print(accuracy)


    def smart_training(self):




        self.model.build_model()

        pixels = np.array(self.model.get_value(self.dev_x.eval())[0])
        pixels = pixels.reshape(28, 28)

        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1);
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        # pixels = self.dev_x[0].eval()
        # pixels = np.array(pixels, dtype='uint8')

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        ax.matshow(pixels, cmap=matplotlib.cm.binary)
        #plt.show()

        print('... training')
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                           # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
        validation_frequency = min(self.number_of_batches_in_train, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        n_epochs = 1
        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(int(self.number_of_batches_in_train)):

                iter = (epoch - 1) * self.number_of_batches_in_train + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = self.model.sgd_step(self.train_x[minibatch_index*self.batch_size:(minibatch_index+1)*self.batch_size].eval(),self.train_y[minibatch_index*self.batch_size:(minibatch_index+1)*self.batch_size].eval())

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.model.test_model(self.dev_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.dev_y[i*self.batch_size:(i+1)*self.batch_size].eval()) for i
                                         in range(int(self.number_of_batches_in_dev))]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, int(self.number_of_batches_in_train),
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter



                if patience <= iter:
                    done_looping = True
                    break
        test_losses = [
                        self.model.test_model(self.dev_x[i * self.batch_size:(i + 1) * self.batch_size].eval(),
                                              self.dev_y[i * self.batch_size:(i + 1) * self.batch_size].eval()) for i
                        in range(int(self.number_of_batches_in_dev))]

        test_loss = np.mean(validation_losses)

        ax2 = fig.add_subplot(1, 3, 2);
        img = self.model.get_layer_output(self.dev_x[0 * self.batch_size:(0 + 1) * self.batch_size].eval())
        pixels = np.array(img[0])
        pixels = pixels.reshape(12, 12)
        ax2.matshow(pixels, cmap=matplotlib.cm.binary)

        ax3 = fig.add_subplot(1, 3, 3);
        img = self.model.get_layer_output(self.dev_x[0 * self.batch_size:(0 + 1) * self.batch_size].eval())
        pixels = np.array(img[1])
        pixels = pixels.reshape(12, 12)
        ax3.matshow(pixels, cmap=matplotlib.cm.binary)

        plt.show()

        fig = plt.figure()
        axx = fig.add_subplot(1, 3, 1);
        axx.matshow(self.model.layer[0].W)

        plt.show
        test_score = 1 - test_loss

        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))


    def load_mnist_data(self,filename):


        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(filename)
        if data_dir == "" and not os.path.isfile(filename):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                filename
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path

        if (not os.path.isfile(filename)) and data_file == 'mnist.pkl.gz':
            from six.moves import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, filename)

        print('... loading data')

        with gzip.open(filename, 'rb') as f:
            train_data, dev_data, test_data = pickle.load(f,encoding='latin1')


        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, theano.tensor.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_data)
        valid_set_x, valid_set_y = shared_dataset(dev_data)
        train_set_x, train_set_y = shared_dataset(train_data)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval


    def save_model(self):
        print("not implemented yet!")

if __name__ == '__main__':
    dr = DigitRecognizer("../data/mnist.pkl.gz")
    #out = dr.step_by_step()
    #dr.train_model()
    dr.smart_training()
    print("The End!")
