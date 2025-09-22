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

print(" ---- definido parâmetros ---- ")
#Pasta para salvar os reseultados
origin_path = "Ex4_graphs"

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x_1,x_2: x_1 ** 2 + x_2 ** 2 -10*(t.cos(t.pi*x_1) + t.cos(t.pi*x_2))

#amostras 
number_samples = 30
xa, xb = -5, 5
t.manual_seed(0)

input_size = 2
output_size = 1

x_train = xa + (xb-xa) * t.rand(number_samples, input_size)
x_val = xa + (xb-xa) * t.rand(number_samples, input_size)

y_train = func(x_train[:,0], x_train[:,1])
y_val = func(x_val[:,0], x_val[:,1])

# neuronios e funções de  ativação a testar
activation_func_list = ("ReLU", "sigmoid", "tanh") # funções de ativação a testar
n_neuronios_inicial = 30 # chute inicial de neuronios
passo_centrado = 1 # passo da diferenças finitas
n_iterations = 10 # número de iterações
n_layers = (4, 6, 8) # número máximo  de layer
n_max_neuronios_per_layer = 1000 # número máximo  de neuronios por camada
cl = [rn.neural_net_interno_2_hidden, rn.neural_net_interno_3_hidden, rn.neural_net_interno_4_hidden, rn.neural_net_interno_5_hidden, rn.neural_net_interno_6_hidden]

print(" ---- treinando e validando a rede ---- ")
with tqdm.tqdm(total=len(activation_func_list)*n_iterations*len(n_layers)) as pbar:

    for activation_func in activation_func_list:
        dict_erros[activation_func] = {}
        '''
        print(f" ---- análise para uma rede com 1 camada oculta com a função de ativação {activation_func} ---- ")
        
        # chute inicial
        for neuronios_1 in (n_neuronios_inicial, n_neuronios_inicial+passo_centrado):
            print(f" - com {neuronios_1} neuronios na 1ª camada")

            #treinando, plotando e definindo erros de extrapolação
            loss_val = tr.processamento_agrupado_2d(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func, x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
            dict_erros[activation_func][neuronios_1] = loss_val
            
            # atualizando barra de progresso
            pbar.update(1)

        #estimando grandiente do erro de extrapolação
        grad = (dict_erros[activation_func][n_neuronios_inicial] - dict_erros[activation_func][n_neuronios_inicial+passo_centrado]) / passo_centrado
        flag = []
        for i in range(n_iterations-1):
            #ajuste do número de neuronios pelo método de newton
            neuronios_old = neuronios_1
            neuronios_1 = int(neuronios_1 - loss_val/grad)

            # controle para não se ter um número excessivo ou negativo de neuronios
            if neuronios_1 >= n_max_neuronios_per_layer:
                if 1 in flag:
                    print(" - loop noticiado")
                    pbar.update(n_iterations - 1 - i)
                    break
                
                flag.append(1)
                neuronios_1 = n_max_neuronios_per_layer
            elif neuronios_1 <= 1:
                if 0 in flag:
                    print(" - loop noticiado")
                    pbar.update(n_iterations - 1 - i)
                    break

                flag.append(0)
                neuronios_1 = 1
            
            print(f" - com {neuronios_1} neuronios na 1ª camada")

            #estimando erro de extrapolação da função
            loss_val_old = loss_val
            loss_val = tr.processamento_agrupado_2d(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func, x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
            dict_erros[activation_func][neuronios_1] = loss_val

            # cálculo do gradiente
            grad = (loss_val - loss_val_old) / (neuronios_1 - neuronios_old)
            
            # atualizando barra de progresso
            pbar.update(1)
        
        '''
        # 2 - max hidden layers
        for n_l in n_layers:
            print(f" ---- análise para uma rede com {n_l} camadas ocultas com a função de ativação {activation_func} ---- ")
            model_class = cl[n_l-2]
            # determinação do chute inicial
            neuronios = [n_neuronios_inicial for i in range(n_l)]

            for i in range(n_iterations):
                grad = []

                #aviso
                text = f" - com {neuronios[0]} neuronios na 1ª camada"
                for j in range(n_l-1):
                    text = text + f"; \n \t {neuronios[j+1]} neuronios na  {j+2}ª camada"
                print(text)
                
                loss_val_old = tr.processamento_agrupado_2d(model_class, neuronios, activation_func, x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
                
                # salvando os erros
                text = str(neuronios[0])
                for j in range(n_l-1):
                    text = text + f",{neuronios[j+1]}"
                dict_erros[activation_func][text] = loss_val_old

                if i == n_iterations -1:
                    pbar.update(1)
                    break

                #rodadndo para determinação da derivada do número de neuronios nas camadas
                for i in range(n_l):
                    temp = []
                    for j in range(n_l):
                        if j == i:
                            temp.append(1)
                        else:
                            temp.append(0)
                    temp = list(np.array(neuronios)+passo_centrado*np.array(temp))
                    #aviso
                    text = f" - com {temp[0]} neuronios na 1ª camada"
                    for j in range(n_l-1):
                        text = text + f"; \n \t {temp[j+1]} neuronios na {j+2}ª camada"
                    print(text)
            
                    #calculando a rede
                    loss_val = tr.processamento_agrupado_2d(model_class, temp, activation_func,  x_train, y_train, x_val, y_val, func, xa, xb, origin_path)
                    
                    # salvando os erros
                    text = str(temp[0])
                    for j in range(n_l-1):
                        text = text + f",{temp[j+1]}"
                    dict_erros[activation_func][text] = loss_val
                
                    #determinado derivada
                    grad.append((loss_val-loss_val_old)/passo_centrado)

                #determinação do próximo chute
                grad = np.array(grad)
                flag = list(np.array(neuronios) - loss_val_old*grad/(np.linalg.norm(grad))**2)
                for i in range(len(flag)):
                    if flag[i] >= n_max_neuronios_per_layer:
                        neuronios[i] = n_max_neuronios_per_layer
                    elif flag[i] <= 1:
                        neuronios[i] = 1
                    else:
                        neuronios[i] = int(flag[i])

                # atualizando barra de progresso
                pbar.update(1)

            print(" ---- salvando erros ---- ")
            dict_erros_str = json.dumps(dict_erros)
            with open(f"Work\{origin_path}\Erros3.txt", "w") as file:
                print(dict_erros_str, file=file)