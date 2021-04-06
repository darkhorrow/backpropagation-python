import numpy as np
import layer

class OutputLayer(layer.Layer):

    def init_w(self, p_random_seed=np.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=0.01,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))
        return self

    def predict(self, p_X):
        #return self._quantization(self._activation(self._net_input(p_X)))
        self.sigma_o = self._activation(self._net_input(p_X))
        return self.sigma_o

    def d_error(self, s_esperada):
        D = np.eye(self.sigma_o.shape[0]) * (self.sigma_o * (1 - self.sigma_o))
        return np.matmul(D, (self.sigma_o - s_esperada))

    def error(self, s_esperada):
        return 0.5 * np.sum((self.sigma_o - s_esperada) ** 2)

    def norma_inf(self, s_seperada):
        return np.max(np.abs(self.sigma_o - s_seperada))

    def saveWeights(self, file):
        np.save(file, self.w)

    def loadWeights(self, file):
        self.w = np.load(file)