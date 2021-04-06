import numpy as np
import layer

class InputLayer(layer.Layer):

    def __init__(self, p_number_neurons=1):
        layer.Layer.__init__(self, p_number_neurons, 1)

    def init_w(self, p_random_seed=np.random.RandomState(None)):
        self.w = np.concatenate((np.zeros((1, self.number_neurons)), np.eye(self.number_neurons)))
        print(self.w)
        return self

    def predict(self, p_X):
        self.sigma_o = self._activation(self._net_input(p_X))
        return self.sigma_o

    def saveWeights(self, file):
        np.save(file, self.w)

    def loadWeights(self, file):
        self.w = np.load(file)