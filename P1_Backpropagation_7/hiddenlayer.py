import numpy as np
import layer


class HiddenLayer(layer.Layer):

    def init_w(self, p_random_seed=np.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=0.01,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))
        return self

    def predict(self, p_X):
        self.sigma_o = self._activation(self._net_input(p_X))
        return self.sigma_o

    def d_error(self):
        return np.eye(self.sigma_o.shape[0]) * (self.sigma_o * (1 - self.sigma_o))

    def saveWeights(self, file):
        np.save(file, self.w)

    def loadWeights(self, file):
        self.w = np.load(file)