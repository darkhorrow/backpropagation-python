from backpropagation import BackPropagation
from data_cleaner import DataCleaner
import numpy as np

def main():
    dc = DataCleaner('datos/Accidentes.xls')
    
    bp = BackPropagation(p_number_iterations=300, p_eta=0.1)
    
    bp.fit(dc.train_x, dc.train_y, dc.valid_x, dc.valid_y, p_number_hidden_layers = 1,
                p_number_neurons_hidden_layers = np.array([dc.train_x.shape[1]]), opt_saveWeights=True, opt_loadWeights=False)
    
    predict_test(bp, dc)
    get_accuracy(bp, dc)

def predict_test(bp, dc):
    print("\nAlertas y etiquetas:")
    for elem_x, elem_y in zip(dc.test_x, dc.test_y):
        result = bp.predict(elem_x)

        color = "[Resultado] "
        if result <= 0.1:
            color += "Verde"
        elif result > 0.1 and result <= 0.5:
            color += "Naranja"
        else:
            color += "Rojo"

        print(color + "\t=>\t[Valor real] " + str(elem_y))

def get_accuracy(bp, dc):
    aciertos = 0
    predictions = []
    for elem_x, elem_y in zip(dc.test_x, dc.test_y):
        result = bp.predict(elem_x)
        predictions.append(result)
        if elem_y == 1 and result >= 0.9:
            aciertos += 1
        elif elem_y == 0 and result <= 0.1:
            aciertos += 1

    print("\nPrecisión: " + str(aciertos / len(dc.test_x)))

if __name__ == '__main__':
    main()