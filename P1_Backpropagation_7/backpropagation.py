import numpy as np
import inputlayer
import hiddenlayer
import outputlayer
import matplotlib.pyplot as plt


class BackPropagation(object):
    """Class BackPropagation:

       Attributes:
         eta.- Learning rate
         number_iterations.-
         ramdon_state.- Random process seed
         input_layer_.-
         hidden_layers_.-
         output_layer_.-
         sse_while_fit_.-

       Methods:
         __init__(p_eta=0.01, p_iterations_number=50, p_ramdon_state=1)
         fit(p_X_training, p_Y_training, p_X_validation, p_Y_validation, p_number_hidden_layers=1, p_number_neurons_hidden_layers=numpy.array([1]))
         predict(p_x) .- Method to predict the output, y

    """

    def __init__(self, p_eta=0.01, p_number_iterations=5000, p_random_state=None):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = np.random.RandomState(p_random_state)
        self.entrenada = False

    def fit(self, p_X_training,
            p_Y_training,
            p_X_validation,
            p_Y_validation,
            p_number_hidden_layers=1,
            p_number_neurons_hidden_layers=np.array([1]),
            opt_saveWeights = False,
            opt_loadWeights = False):

        self.input_layer_ = inputlayer.InputLayer(p_X_training.shape[1])
        self.hidden_layers_ = []

        metadatos_red = []
        metadatos_red.append(p_X_training.shape[1])

        for v_layer in range(p_number_hidden_layers):
            if v_layer == 0:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   self.input_layer_.number_neurons))
            else:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   p_number_neurons_hidden_layers[v_layer - 1]))

            metadatos_red.append(p_number_neurons_hidden_layers[v_layer])

        self.output_layer_ = outputlayer.OutputLayer(p_Y_training.shape[1],
                                                     self.hidden_layers_[
                                                         self.hidden_layers_.__len__() - 1].number_neurons)

        metadatos_red.append(p_Y_training.shape[1])
        self.metadatos_red = np.array(metadatos_red)

        if opt_loadWeights:
            try:
                self.__loadWeights(self.number_iterations)
                self.entrenada = True
            except FileNotFoundError:
                print("No se han encotrado los ficheros de carga.\n"
                      "Desactive la opción opt_loadWeights.\n"
                      "Si lo desea puede crear una copia de los pesos para el número de iteraciones establecido"
                      " mediante la activación de la opción opt_saveWeights.")
                exit(4)
        else:
            self.input_layer_.init_w(self.random_seed)
            for v_hidden_layer in self.hidden_layers_:
                v_hidden_layer.init_w(self.random_seed)
            self.output_layer_.init_w(self.random_seed)

            # ...

            self.entrenada = True

            costos = []
            validation = []
            progress = 1
            print("Entrenando...")
            for i in range(self.number_iterations):

                error_acumulado = 0

                for k in range(len(p_X_training)):
                    delta = []
                    delta_pesos = []

                    self.predict(p_X_training[k])

                    delta.append(np.array([self.output_layer_.d_error(p_Y_training[k])]))

                    o_prima = np.array([np.insert(self.hidden_layers_[-1].get_sigma(), 0, 1)])
                    delta_pesos.append(np.array(self.eta * np.matmul(delta[-1].T, o_prima)).astype(
                        'float64'))

                    for j in range(len(self.hidden_layers_) - 1, -1, -1):

                        if j == len(self.hidden_layers_) - 1:
                            delta.append(np.array(np.matmul(
                                np.matmul(self.hidden_layers_[j].d_error(), self.output_layer_.w[1:]), delta[-1].T)))

                        else:
                            delta.append(np.array(np.matmul(
                                np.matmul(self.hidden_layers_[j].d_error(), self.hidden_layers_[j + 1].w[1:]), delta[-1])))

                        if j > 0:

                            o_prima = np.array([np.insert(self.hidden_layers_[j - 1].get_sigma(), 0, 1)])
                            delta_pesos.append(np.array(self.eta * np.matmul(delta[-1], o_prima)).astype('float64'))

                        else:

                            o_prima = np.array([np.insert(self.input_layer_.get_sigma(), 0, 1)])
                            delta_pesos.append(np.array(
                                self.eta * np.matmul(delta[-1], o_prima)).astype('float64'))

                    delta_pesos = np.array(delta_pesos)
                    self.output_layer_.w -= delta_pesos[0].T

                    m = 1
                    for capa_oculta in reversed(self.hidden_layers_):
                        capa_oculta.w -= delta_pesos[m].T
                        m += 1

                    error_acumulado += self.output_layer_.norma_inf(p_Y_training[k])

                    # print(self.output_layer_.error(p_Y_training[k]))

                error_promedio_train = error_acumulado / len(p_X_training)

                if error_promedio_train < 0.1:
                    self.eta *= 0.99995
                    
                costos.append(error_promedio_train)

                error_acumulado = 0

                for val in range(len(p_X_validation)):
                    self.predict(p_X_validation[val])
                    error_acumulado += self.output_layer_.norma_inf(p_Y_validation[val])

                error_promedio_validate = error_acumulado / len(p_X_validation)
                
                print("\rProgreso: " + str(progress) + "/" + str(self.number_iterations) + "\tloss: " + str(error_promedio_train) + "\tval: " + str(error_promedio_validate), end='')

                validation.append(error_promedio_validate)
                
                progress += 1

            # ...

            self.costos = np.array(costos)
            self.validation = np.array(validation)

            if(opt_saveWeights):
                self.__saveWeights(self.number_iterations)

        plt.plot(self.costos, 'b-', label="Error")
        plt.plot(self.validation, 'r-', label="Validacion")
        plt.legend()
        plt.title("Evolución del error")
        plt.xlabel("Iteraciones")
        plt.ylabel("Error de ajuste")
        plt.show()

        return self

    def predict(self, p_X):
        if not self.entrenada:
            print("La red aún no ha sido entrenada.")
            exit(1)
        else:
            v_Y_input_layer_ = self.input_layer_.predict(p_X)
            v_X_hidden_layer_ = v_Y_input_layer_
            for v_hiddenlayer in self.hidden_layers_:
                v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
                v_X_hidden_layer_ = v_Y_hidden_layer_
            v_X_output_layer_ = v_X_hidden_layer_
            v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
            return v_Y_output_layer_

    def error(self, entradas, s_esperada):
        if not self.entrenada:
            print("La red aún no ha sido entrenada.")
            exit(2)
        else:
            self.predict(entradas)
            return self.output_layer_.error(s_esperada)

    def __saveWeights(self, num_iters):
        i = 0
        root_name = "pesos/weights_" + str(num_iters) + "_" + np.array2string(self.metadatos_red) + "_"

        self.input_layer_.saveWeights(root_name + str(i) + ".npy")

        for v_hiddenlayer in self.hidden_layers_:
            i += 1
            v_hiddenlayer.saveWeights(root_name + str(i) + ".npy")

        i += 1
        self.output_layer_.saveWeights(root_name + str(i) + ".npy")

        np.save(root_name + "costos.npy", self.costos)
        np.save(root_name + "validation.npy", self.validation)
        np.save(root_name + "metadatos.npy", self.metadatos_red)

    def __loadWeights(self, num_iters):
        i = 0
        root_name = "pesos/weights_" + str(num_iters) + "_" + np.array2string(self.metadatos_red) + "_"

        aux_metadatos = np.load(root_name + "metadatos.npy")

        if(np.array_equal(aux_metadatos, self.metadatos_red)):

            self.input_layer_.loadWeights(root_name + str(i) + ".npy")
            for v_hiddenlayer in self.hidden_layers_:
                i += 1
                v_hiddenlayer.loadWeights(root_name + str(i) + ".npy")

            i += 1
            self.output_layer_.loadWeights(root_name + str(i) + ".npy")

            self.costos = np.load(root_name + "costos.npy")
            self.validation = np.load(root_name + "validation.npy")

        else:
            print("No se encotró una red con la estructura solicitada.\n"
                  "Desactive la opción opt_loadWeights.\n"
                  "Si lo desea puede crear una copia de los pesos para el número de iteraciones establecido"
                  " mediante la activación de la opción opt_saveWeights.")
            exit(3)