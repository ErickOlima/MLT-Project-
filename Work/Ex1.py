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
def plotagem_1d_modelo(xf, func, x_true, y_pred, y_true, loss, func_label, path):
    '''
    xf: intervalo onde plotar a função
    func: função exata
    x_true: valores x de trainamento
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
    plt.close(fig)

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

#valores de validação
xf = np.linspace(-6, 6, 12)

#erro de função
error_func = t.nn.MSELoss()

# neuronios e funções de  ativação a testar
activation_func_list = ("ReLU", "sigmoid", "tanh")
neuronios_list = (5, 10, 15, 20, 25, 30)

print(" ---- treinando e validando a rede ---- ")
with tqdm.tqdm(total=len(activation_func_list) * len(neuronios_list)*(1+len(neuronios_list)*(1+len(neuronios_list)))) as pbar:
    
    # error_func = lambda y_true, y_pred = np.sum((func(xf)-y_pred.detach().numpy())**2)
    for activation_func in activation_func_list:
        dict_erros[activation_func] = {}
        begin = tm.time()

        # 1 hidden layer
        for neuronios_1 in (5, 10, 15, 20, 25, 30):
            print(f" - com {neuronios_1} neuronios na 1ª camada")

            #treinando, plotando e definindo erros de extrapolação
            error_measure = tr.processamento_agrupado(rn.neural_net_interno_1_hidden, [neuronios_1], activation_func, 
                                                        x_train, y_train, x_val, y_val, xf, func, error_func)
            dict_erros[activation_func][neuronios_1] = error_measure

            # atualizando barra de progresso
            pbar.update(1)
           
        # 2 hidden layer
        for neuronios_1 in neuronios_list:
            for neuronios_2 in neuronios_list:
                neuronios = [neuronios_1, neuronios_2]
                print(f" - com {neuronios[0]} neuronios na 1ª camada \n \t e {neuronios[1]} neuronios na 2ª camada")
                
                error_measure = tr.processamento_agrupado(rn.neural_net_interno_2_hidden, neuronios, activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
                dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]}"] = error_measure

                pbar.update(1)

        # 3 hidden layer
        for neuronios_1 in neuronios_list:
            for neuronios_2 in neuronios_list:
                for neuronios_3 in neuronios_list:
                    print(f" - com {neuronios[0]} neuronios na 1ª camada; \n \t {neuronios[1]} neuronios na 2ª camada; \n \t e {neuronios[2]} neuronios na 3ª camada")
                    
                    error_measure_old = tr.processamento_agrupado(rn.neural_net_interno_3_hidden, neuronios, activation_func, x_train, y_train, x_val, y_val, xf, func, error_func)
                    dict_erros[activation_func][f"{neuronios[0]},{neuronios[1]},{neuronios[2]}"] = error_measure_old
                    
                    pbar.update(1)

        for erros in dict_erros[activation_func].items():
            print(f"Função de Ativação: {activation_func}, Neurônios: {erros[0]}, Erro: {erros[1]}")
        
        print(f"Tempo para função de ativação {activation_func}: {tm.time()-begin:.2f} segundos")

print(" ---- salvando erros ---- ")
dict_erros = json.dumps(dict_erros)
with open("Work\Ex1_graphs\Erros.txt", "w") as file:
    print(dict_erros, file=file)