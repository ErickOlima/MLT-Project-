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
import sys

def processamento_agrupado_2d(model_class, neuronios, activation_func, x_train, y_train, x_val, y_val,
                            func, xa, xb):
    neuronios_tot = [2] + neuronios + [1]

    #instanciaando a rede
    model = model_class(neuronios_tot, activation_func, zerar_seed=1)

    #realizando o treinamento
    loss_train, loss_val, min_loss_val, max_loss_val, savestate = model.treino_validacao(x_train, y_train, x_val, y_val)

    # recuperando estado ótimo do modelo
    model.load_state_dict(savestate)
    
    # plotando os dados
    path = f"Work\Ex2_graphs\{activation_func}"
    for i in range(len(neuronios)):
        path = path + f"_hidden{i+1}_{neuronios[i]}"

    fig = rn.plt_plot_general()
    fig.plot_3d_surface(model, xa, xb, func)
    fig.plot_erros(loss_train, loss_val)
    fig.plot_training_points(x_train, x_val)
    fig.save_fig(path)
    fig.close()

    #retornando  erro da avaliação
    return min_loss_val

print(" ---- definido parâmetros ---- ")

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x_1,x_2: 3*(1-x_1)**2*np.exp(-x_1**2 - (x_2 + 1)**2) - 10*(x_1/5 - x_1**3 - x_2**5)*np.exp(-x_1**2 - x_2**2) - 1/3*np.exp(-(x_1 + 1)**2 - x_2**2)
#amostras 
number_samples = 20
xa, xb = -4, 4
t.manual_seed(0)

input_size = 2
output_size = 1

x_train = xa + (xb-xa) * t.rand(number_samples, input_size)
x_val = xa + (xb-xa) * t.rand(number_samples, input_size)

y_train = func(x_train[:,0], x_train[:,1])
y_val = func(x_val[:,0], x_val[:,1])

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

    ''' # 1 hidden layer
    for neuronios_1 in neuronios_list:
        loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func,
                                               x_train, y_train, x_val, y_val, func, xa, xb)
        dict_erros[activation_func][neuronios_1] = loss_val_min
        
    # 2 hidden layer
    for neuronios_1 in neuronios_list:
        for neuronios_2 in neuronios_list:
                if neuronios_1 == neuronios_2:
                    loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_2_hidden, [neuronios_1, neuronios_2], activation_func, x_train, y_train, x_val, y_val, func, xa, xb)
                    dict_erros[activation_func][f"{neuronios_1},{neuronios_2}"] = loss_val_min 
                    '''
    # 3 hidden layer
    for neuronios_1 in neuronios_list:
        for neuronios_2 in neuronios_list:
            for neuronios_3 in neuronios_list:
                if neuronios_1 == neuronios_2 == neuronios_3 or neuronios_1 > neuronios_2 > neuronios_3 or neuronios_1 < neuronios_2 < neuronios_3 or neuronios_1 < neuronios_2 > neuronios_3 or neuronios_1 > neuronios_2 < neuronios_3:    
                    loss_val_min = tr.processamento_agrupado_2d(rn.neural_net_interno_3_hidden, [neuronios_1, neuronios_2, neuronios_3], activation_func, x_train, y_train, x_val, y_val, func, xa, xb)
                    dict_erros[activation_func][f"{neuronios_1},{neuronios_2},{neuronios_3}"] = loss_val_min 
                    sys.exit(0)
            
    for erros in dict_erros[activation_func].items():
        print(f"Função de Ativação: {activation_func}, Neurônios: {erros[0]}, Erro: {erros[1]}")
    
    print(f"Tempo para função de ativação {activation_func}: {tm.time()-begin:.2f} segundos")

print(" ---- salvando erros ---- ")
dict_erros = json.dumps(dict_erros)
with open("Work\Ex2_graphs\Erros.txt", "w") as file:
    print(dict_erros, file=file)
