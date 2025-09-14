import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt

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

#função
func = lambda x: x ** 3
#amostras 
number_samples = 20
xa, xb = -4, 4
#t.manual_seed(0)
x_true = xa + (xb-xa) * t.rand(number_samples, 1)
y_true = func(x_true)

model = rn.neural_net_interno_1_hidden([1, 20, 1]) # uma camada interna de 20]
loss = model.treino(x_true, y_true)

xf = np.linspace(-6, 6, 12)
y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32))

error_measure = np.sum((func(xf)-y_pred.detach().numpy())**2)
print(error_measure)
'''
fig, (ax, ax2) = plt.subplots(2,1, figsize = (7, 10))
ax.plot(xf, func(xf), label = "y = x³")
ax.plot(xf, y_pred.detach().numpy(), label = "dados previstos")
ax.scatter(x_true, y_true, label = "valores de treinamento")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax2.set_title("Evolução da perda")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss")
ax2.plot(range(4000), loss)
fig.savefig("Work\Ex1_graphs\Relu_hidden_1_20")
'''
plotagem_1d_modelo(xf, func, y_pred, y_true, loss, "y = x³", "Work\Ex1_graphs\Relu_hidden1_20")