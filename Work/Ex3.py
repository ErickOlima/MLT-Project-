print(" ---- importando dependências ---- ")
import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import Rede_neural as rn
import time as tm



print(" ---- definido parâmetros ---- ")

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x_1,x_2: t.sin(x_1+ t.cos(x_2))
#amostras 
number_samples = 15
xa, xb = -3, 3
#t.manual_seed(0)

input_size = 2
output_size = 1

x_true = xa + (xb-xa) * t.rand(number_samples, input_size)
y_true = func(x_true[:,0], x_true[:,1])

#valores de validação
xf = np.linspace(-3, 3, 12)

#erro de função
error_func = t.nn.MSELoss()

# neuronios e funções de  ativação a testar
activation_func_list = ("ReLU", "sigmoid", "tanh")
neuronios_list = (5, 15,30)

print(" ---- treinando e validando a rede ---- ")
#with tqdm.tqdm(total=len(activation_func_list) + len(neuronios_list)*(1+len(neuronios_list)*(1+len(neuronios_list)))) as pbar:
    
# error_func = lambda y_true, y_pred = np.sum((func(xf)-y_pred.detach().numpy())**2)
for activation_func in activation_func_list:
    dict_erros[activation_func] = {}
    begin = tm.time()

    # 1 hidden layer
    for neuronios_1 in neuronios_list:
        model = rn.neural_net_interno_1_hidden([input_size, neuronios_1, output_size], activation_func, zerar_seed=1) # uma camada interna de 20]
        loss = model.treino(x_true, y_true)

        y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

        error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
        dict_erros[activation_func][neuronios_1] = error_measure

        rn.plt_3d_surface(xf, func, x_true, y_pred, y_true, loss, "y = sin(x1 + cos(x2))",
                            f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}")

   
    # 2 hidden layer
    for neuronios_1 in neuronios_list:
        for neuronios_2 in neuronios_list:
                if neuronios_1 == neuronios_2:
                    model = rn.neural_net_interno_2_hidden([input_size, neuronios_1, neuronios_2, output_size], activation_func) # uma camada interna de 20]
                    loss = model.treino(x_true, y_true)

                    y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

                    error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
                    dict_erros[activation_func][f"{neuronios_1},{neuronios_2}"] = error_measure

                    rn.plt_3d_surface(xf, func, x_true, y_pred, y_true, loss, "y = sin(x1 + cos(x2))",
                                    f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}_hidden2_{neuronios_2}")
                    
                


    # 3 hidden layer
    for neuronios_1 in neuronios_list:
        for neuronios_2 in neuronios_list:
            for neuronios_3 in neuronios_list:
                if neuronios_1 == neuronios_2 == neuronios_3 or neuronios_1 > neuronios_2 > neuronios_3 or neuronios_1 < neuronios_2 < neuronios_3 or neuronios_1 < neuronios_2 > neuronios_3 or neuronios_1 > neuronios_2 < neuronios_3:    
                    model = rn.neural_net_interno_3_hidden([input_size, neuronios_1, neuronios_2, neuronios_3, output_size], activation_func) # uma camada interna de 20]
                    loss = model.treino(x_true, y_true)

                    y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()

                    error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()
                    dict_erros[activation_func][f"{neuronios_1},{neuronios_2},{neuronios_3}"] = error_measure
                    
                    rn.plt_3d_surface(xf, func, x_true, y_pred, y_true, loss, "y = sin(x1 + cos(x2))",
                                f"Work\Ex1_graphs\{activation_func}_hidden1_{neuronios_1}_hidden2_{neuronios_2}_hidden3_{neuronios_3}")


                
    for erros in dict_erros[activation_func].items():
        print(f"Função de Ativação: {activation_func}, Neurônios: {erros[0]}, Erro: {erros[1]}")
    
    print(f"Tempo para função de ativação {activation_func}: {tm.time()-begin:.2f} segundos")

#print(" ---- salvando erros ---- ")
#dict_erros = json.dumps(dict_erros)
#with open("Work\Ex1_graphs\Erros.txt", "w") as file:
#    print(dict_erros, file=file)