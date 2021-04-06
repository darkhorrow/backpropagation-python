import abc
import numpy as np

class Layer(object):
    """Class Layer:

    Attributes:
        number_neurons.-
        number_inputs_each_neuron.-
        w.-

    Methods:
         __init__(p_number_neurons, p_number_inputs, p_random_state)
         init_w()
         _net_input(p_X)
         _activation(p_net_input)
         _quantization(p_activation)
         predict(p_X)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, p_number_neurons=1, p_number_inputs_each_neuron=1):
        self.number_neurons = p_number_neurons
        self.number_inputs_each_neuron = p_number_inputs_each_neuron
        self.sigma_o = np.array([])

    @abc.abstractmethod
    def init_w(self, p_random_seed=np.random.RandomState(None)):
        pass

    def _net_input(self, p_X):
        #print("Pesos: " + str(self.w))
        return np.matmul(p_X, self.w[1:, :]) + self.w[0, :]

    def _activation(self, p_net_input):
        #print("Net input: " + str(p_net_input))
        return self._sigmoid(p_net_input)

    #def _quantization(self, p_activation):
    #    return numpy.where(p_activation >= 0.0, 1, -1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x.astype(float)))

    @abc.abstractmethod
    def d_error(self):
        pass

    def get_sigma(self):
        return np.array([self.sigma_o])

    @abc.abstractmethod
    def saveWeights(self, file):
        pass

    @abc.abstractmethod
    def loadWeights(self, file):
        pass

    @abc.abstractmethod
    def predict(self, p_X):
        pass
