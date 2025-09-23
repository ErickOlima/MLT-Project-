import numpy as np
import matplotlib.pyplot as plt
import torch as t
import Rede_neural as rn

def processamento_agrupado_2d(model_class, neuronios, activation_func, x_train, y_train, x_val, y_val,
                            func, xa, xb, origin_path):
    neuronios_tot = [2] + neuronios + [1]

    #instanciaando a rede
    model = model_class(neuronios_tot, activation_func, zerar_seed=1)

    #realizando o treinamento
    loss_train, loss_val, min_loss_val, max_loss_val, savestate = model.treino_validacao(x_train, y_train, x_val, y_val)

    # recuperando estado ótimo do modelo
    model.load_state_dict(savestate)
    
    # plotando os dados
    path = f"Work\{origin_path}\{activation_func}"
    for i in range(len(neuronios)):
        path = path + f"_hidden{i+1}_{neuronios[i]}"

    fig = rn.plt_plot_general()
    fig.plot_3d_surface(model, xa, xb, func)
    fig.plot_erros(loss_train, loss_val)
    fig.plot_training_points(x_train, x_val)
    fig.show(path)

    #retornando  erro da avaliação
    return min_loss_val

def processamento_agrupado(model_class, neuronios, activation_func, x_train, y_train, x_val, y_val,
                            xf, func, error_func):
    neuronios_tot = [1] + neuronios + [1]

    #instanciaando a rede
    model = model_class(neuronios_tot, activation_func, zerar_seed=1)

    #realizando o treinamento
    loss_train, loss_val, mini, maxi, savestate = model.treino_validacao(x_train, y_train, x_val, y_val)

    # recuperando estado ótimo do modelo
    model.load_state_dict(savestate)
    
    #determinando valores finais após treinamento
    y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32)).squeeze()
    
    #estimando erro de extrapolação da função
    error_measure = error_func(t.tensor(func(xf), dtype= t.float64), y_pred).item()

    #plotando a resultados
    text = f"Work\Ex1_graphs\{activation_func}"
    for i in range(len(neuronios)):
        text = text + f"_hidden{i+1}_{neuronios[i]}"
    rn.plotagem_1d_modelo_show(xf, func, y_pred, x_train, y_train, x_val, y_val, loss_train, loss_val, "y = x³",text)
    
    return error_measure