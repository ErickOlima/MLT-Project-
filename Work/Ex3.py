print(" ---- importando dependências ---- ")
import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import Rede_neural as rn
import time as tm
import Treinamenro_de_rede as tr


print(" ---- definido parâmetros ---- ")
#Pasta para salvar os reseultados
origin_path = "Ex3_graphs"

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x_1,x_2: t.sin(x_1+ t.cos(x_2))

#amostras 
number_samples = 15
xa, xb = -3, 3
t.manual_seed(0)

input_size = 2
output_size = 1

x_train = xa + (xb-xa) * t.rand(number_samples, input_size)
x_val = xa + (xb-xa) * t.rand(number_samples, input_size)

y_train = func(x_train[:,0], x_train[:,1])
y_val = func(x_val[:,0], x_val[:,1])

# neuronios e funções de  ativação a testar
activation_func_list = ("ReLU", "sigmoid", "tanh")
neuronios_list = (5, 15,30)

print(" ---- treinando e validando a rede ---- ")
with tqdm.tqdm(total=len(activation_func_list)*len(neuronios_list)*(1+len(neuronios_list)*(1+len(neuronios_list)))) as pbar:
    
    for activation_func in activation_func_list:
        dict_erros[activation_func] = {}
        begin = tm.time()

        # 1 hidden layer
        for neuronios_1 in neuronios_list:
            loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func,
                                                x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
            dict_erros[activation_func][neuronios_1] = loss_val_min
            pbar.update(1)

        # 2 hidden layers
        for neuronios_1 in neuronios_list:
            for neuronios_2 in neuronios_list:
                    if neuronios_1 == neuronios_2:
                        loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_2_hidden, [neuronios_1, neuronios_2], activation_func, x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
                        dict_erros[activation_func][f"{neuronios_1},{neuronios_2}"] = loss_val_min 
                    pbar.update(1)
                        
        # 3 hidden layers
        for neuronios_1 in neuronios_list:
            for neuronios_2 in neuronios_list:
                for neuronios_3 in neuronios_list:
                    if neuronios_1 == neuronios_2 == neuronios_3 or neuronios_1 > neuronios_2 > neuronios_3 or neuronios_1 < neuronios_2 < neuronios_3 or neuronios_1 < neuronios_2 > neuronios_3 or neuronios_1 > neuronios_2 < neuronios_3:    
                        loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_3_hidden, [neuronios_1, neuronios_2, neuronios_3], activation_func, x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
                        dict_erros[activation_func][f"{neuronios_1},{neuronios_2},{neuronios_3}"] = loss_val_min 
                    pbar.update(1)
                
        for erros in dict_erros[activation_func].items():
            #if erros[1] == min(dict_erros[activation_func].values()):
                print(f"Função de Ativação: {activation_func}, Neurônios: {erros[0]}, Erro: {erros[1]}")
        
        print(f"Tempo para função de ativação {activation_func}: {tm.time()-begin:.2f} segundos")

    print(" ---- salvando erros ---- ")
    dict_erros = json.dumps(dict_erros)
    with open(f"Work\{origin_path}\Erros.txt", "w") as file:
        print(dict_erros, file=file)
