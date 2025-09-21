print(" ---- importando dependências ---- ")
import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import Rede_neural as rn
import Treinamenro_de_rede as tr

print(" ---- definido parâmetros ---- ")

#dicionário com os erros de diversas situações
dict_erros = {}

#função
func = lambda x: x ** 3
#amostras 
number_samples = 20
xa, xb = -4, 4
t.manual_seed(0)
x_train = xa + (xb-xa) * t.rand(number_samples, 1)
y_train = func(x_train)

x_val = xa + (xb-xa) * t.rand(number_samples, 1)
y_val = func(x_val)

#valores de extrapolação
xf = np.linspace(-6, 6, 12)

#erro de função
error_func = t.nn.MSELoss()
# error_func = lambda y_true, y_pred = np.sum((func(xf)-y_pred.detach().numpy())**2)

# neuronios e funções de  ativação a testar
activation_func_list = ("ReLU", "sigmoid", "tanh")
n_neuronios_inicial = 10
passo_centrado = 20
n_iterations = 2
n_max_layers = 3
n_max_neuronios_per_layer = 1000

print(" ---- treinando e validando a rede ---- ")
with tqdm.tqdm(total=len(activation_func_list)*n_iterations*n_max_layers) as pbar:

    for activation_func in activation_func_list:
        dict_erros[activation_func] = {}

        print(f" ---- análise para uma rede com 1 camada oculta com a função de ativação {activation_func} ---- ")
        
        # chute inicial
        for neuronios_1 in (n_neuronios_inicial, n_neuronios_inicial+passo_centrado):
            print(f" - com {neuronios_1} neuronios na 1ª camada")

            #treinando, plotando e definindo erros de extrapolação
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func, 
                                                      x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][neuronios_1] = error_measure
            
            # atualizando barra de progresso
            pbar.update(1)

        #estimando grandiente do erro de extrapolação
        grad = (dict_erros[activation_func][n_neuronios_inicial] - dict_erros[activation_func][n_neuronios_inicial+passo_centrado]) / passo_centrado
        flag = []
        for i in range(n_iterations-1):
            #ajuste do número de neuronios pelo método de newton
            neuronios_old = neuronios_1
            neuronios_1 = int(neuronios_1 - error_measure/grad)

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
            error_old = error_measure
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][neuronios_1] = error_measure

            # cálculo do gradiente
            grad = (error_measure - error_old) / (neuronios_1 - neuronios_old)
            
            # atualizando barra de progresso
            pbar.update(1)
        
        # 2 hidden layer
        # determinação do chute inicial
        neuronios = [n_neuronios_inicial, n_neuronios_inicial]

        for i in range(n_iterations):
            grad = []
            print(f" - com {neuronios[0]} neuronios na 1ª camada \n \t e {neuronios[1]} neuronios na 2ª camada")
            
            error_measure_old = tr.processamento_agrupado(rn.neural_net_interno_2_hidden, neuronios, activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]}"] = error_measure_old
            
            #rodadndo para determinação da derivada do número de neuronios na 1ª camada
            print(f" - com {neuronios[0] + passo_centrado} neuronios na 1ª camada \n \t e {neuronios[1]} neuronios na 2ª camada")
        
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_2_hidden, [neuronios[0] + passo_centrado, neuronios[1]], activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]+passo_centrado},{neuronios[1]}"] = error_measure
            
            #determinado derivada
            grad.append((error_measure-error_measure_old)/passo_centrado)

            #rodadndo para determinação da derivada do número de neuronios na 2ª camada
            print(f" - com {neuronios[0]} neuronios na 1ª camada \n \t e {neuronios[1] + passo_centrado} neuronios na 2ª camada")
            
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_2_hidden, [neuronios[0], neuronios[1] + passo_centrado], activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]+passo_centrado}"] = error_measure

            #determinado derivada
            grad.append((error_measure-error_measure_old)/passo_centrado)
            grad = np.array(grad)

            #determinação do próximo chute
            flag = list(np.array(neuronios) - error_measure_old*grad/(np.linalg.norm(grad))**2)
            for i in range(len(flag)):
                if flag[i] >= n_max_neuronios_per_layer:
                    neuronios[i] = n_max_neuronios_per_layer
                elif flag[i] <= 1:
                    neuronios[i] = 1
                else:
                    neuronios[i] = int(flag[i])

            # atualizando barra de progresso
            pbar.update(1)
        
        # 3 hidden layer
        # determinação do chute inicial
        neuronios = [n_neuronios_inicial, n_neuronios_inicial, n_neuronios_inicial]

        for i in range(n_iterations):
            grad = []
            print(f" - com {neuronios[0]} neuronios na 1ª camada; \n \t {neuronios[1]} neuronios na 2ª camada; \n \t e {neuronios[2]} neuronios na 3ª camada")
            
            error_measure_old = tr.processamento_agrupado(rn.neural_net_interno_3_hidden, neuronios, activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]},{neuronios[2]}"] = error_measure_old
            
            #rodadndo para determinação da derivada do número de neuronios na 1ª camada
            print(f" - com {neuronios[0]+passo_centrado} neuronios na 1ª camada; \n \t {neuronios[1]} neuronios na 2ª camada; \n \t e {neuronios[2]} neuronios na 3ª camada")
        
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_3_hidden, [neuronios[0] + passo_centrado, neuronios[1], neuronios[2]], activation_func,  x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]+passo_centrado},{neuronios[1]},{neuronios[2]}"] = error_measure
            
            #determinado derivada
            grad.append((error_measure-error_measure_old)/passo_centrado)

            #rodadndo para determinação da derivada do número de neuronios na 2ª camada
            print(f" - com {neuronios[0]} neuronios na 1ª camada; \n \t {neuronios[1] + passo_centrado} neuronios na 2ª camada; \n \t e {neuronios[2]} neuronios na 3ª camada")
        
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_3_hidden, [neuronios[0], neuronios[1] + passo_centrado, neuronios[2]], activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]+passo_centrado},{neuronios[2]}"] = error_measure

            #determinado derivada
            grad.append((error_measure-error_measure_old)/passo_centrado)

            #rodadndo para determinação da derivada do número de neuronios na 3ª camada
            print(f" - com {neuronios[0]} neuronios na 1ª camada; \n \t {neuronios[1]} neuronios na 2ª camada; \n \t e {neuronios[2] + passo_centrado} neuronios na 3ª camada")
        
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_3_hidden, [neuronios[0], neuronios[1], neuronios[2] + passo_centrado], activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]},{neuronios[2]+ passo_centrado}"] = error_measure

            #determinado derivada
            grad.append((error_measure-error_measure_old)/passo_centrado)
            grad = np.array(grad)

            #determinação do próximo chute
            flag = list(np.array(neuronios) - error_measure_old*grad/(np.linalg.norm(grad))**2)
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
dict_erros = json.dumps(dict_erros)
with open("Work\Ex1_graphs\Erros.txt", "w") as file:
    print(dict_erros, file=file)