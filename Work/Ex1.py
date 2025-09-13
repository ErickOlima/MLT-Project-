import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt

import Rede_neural as rn

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
a = t.tensor(xf, dtype = t.float32).T
y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32))

plt.plot(xf, func(xf), label = "y = x³")
plt.plot(xf, y_pred.detach().numpy(), label = "dados previstos")
plt.scatter(x_true, y_true, label = "valores de treinamento")
plt.legend()
plt.show()