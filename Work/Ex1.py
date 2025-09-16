import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt
import json

import Rede_neural as rn

def plotagem_1d_modelo(xf, func, y_pred, y_true, loss, func_label, path):
    '''
    xf: intervalo onde plotar a função
    func: função exata
    y_pred: valores preditos pelo modelo
    y_true: valores de treinamento
    loss: vetor com as perdas
    func_label: texto a associar com função exata
    path: caminho e nome do arquivo para salvar
    '''
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (7, 10))
    ax.plot(xf, func(xf), label = func_label)
    ax.plot(xf, y_pred.detach().numpy(), label = "dados previstos")
    ax.scatter(x_true, y_true, label = "valores de treinamento")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax2.set_title("Evolução da perda")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.plot(range(4000), loss)
    fig.savefig(path)

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x: x ** 3
#amostras 
number_samples = 20
xa, xb = -4, 4
#t.manual_seed(0)
x_true = xa + (xb-xa) * t.rand(number_samples, 1)
y_true = func(x_true)

#valores de validação
xf = np.linspace(-6, 6, 12)

#erro de função
error_func = t.nn.MSELoss()
# error_func = lambda y_true, y_pred = np.sum((func(xf)-y_pred.detach().numpy())**2)
for activation_func in ("ReLU", "sigmoid", "tanh"):
    dict_erros[activation_func] = {}

    # 1 hidden layer
    for neuronios_1 in (5, 10, 15, 20, 25, 30):
        model = rn.neural_net_interno_1_hidden([1, neuronios_1, 1], activation_func) # uma camada interna de 20]
        loss = model.treino(x_true, y_true)

        y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

        error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
        dict_erros[activation_func][neuronios_1] = error_measure

        plotagem_1d_modelo(xf, func, y_pred, y_true, loss, "y = x³",
                            f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}")

    # 2 hidden layer
    for neuronios_1 in (5, 10, 15, 20, 25, 30):
        for neuronios_2 in (5, 10, 15, 20, 25, 30):
            break
            model = rn.neural_net_interno_2_hidden([1, neuronios_1, neuronios_2, 1], activation_func) # uma camada interna de 20]
            loss = model.treino(x_true, y_true)

            y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

            error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
            dict_erros[activation_func][f"{neuronios_1},{neuronios_2}"] = error_measure

            plotagem_1d_modelo(xf, func, y_pred, y_true, loss, "y = x³",
                                f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}_hidden2_{neuronios_2}")

    # 3 hidden layer
    for neuronios_1 in (5, 10, 15, 20, 25, 30):
        for neuronios_2 in (5, 10, 15, 20, 25, 30):
            for neuronios_3 in (5, 10, 15, 20, 25, 30):
                model = rn.neural_net_interno_3_hidden([1, neuronios_1, neuronios_2, neuronios_3, 1], activation_func) # uma camada interna de 20]
                loss = model.treino(x_true, y_true)

                y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

                error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
                dict_erros[activation_func][f"{neuronios_1},{neuronios_2},{neuronios_3}"] = error_measure

                plotagem_1d_modelo(xf, func, y_pred, y_true, loss, "y = x³",
                                    f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}_hidden2_{neuronios_2}_hidden3_{neuronios_3}")

dict_erros = json.dumps(dict_erros)
with open("Work\Ex1_graphs\Erros.txt", "w") as file:
    print(dict_erros, file=file)