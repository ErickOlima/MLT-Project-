import torch as t
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class neural_net_interno(nn.Module):
    def __init__(self, activation_func = "ReLU"):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super(neural_net_interno, self).__init__()

        if activation_func == "ReLU":
            self.func = nn.ReLU()
        elif activation_func == "sigmoid":
            self.func = nn.Sigmoid()
        elif activation_func == "tanh":
            self.func = nn.Tanh()
        else: 
            raise NameError("função de ativação não encontrada ou não definida")
    
    def treino(self, x_true, y_true, learning_rate=0.01, number_epochs=4000, loss_func=nn.MSELoss(),
                  tipo = "Batch", optimize = "Adam", zerar_seed=False, Mini_batch_size = None, seed =0):
        train_loss_sum_vec = []

        if zerar_seed:
            t.manual_seed(seed)

        if optimize == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr = learning_rate)

            for epoch in range(number_epochs):
                self.train() # permite ao programa treinar
                train_loss_sum = 0

                if tipo == "Batch":
                    # foward pass
                    y_pred = self.foward(x_true).squeeze()
                    loss = loss_func(y_pred, y_true.squeeze())
                    train_loss_sum += loss.item() # tensor.item() retorna o valor nummerico do tensor caso seja 1x1

                    #backward
                    optimizer.zero_grad() #zera os valores .grad calculados
                    loss.backward() # cálculo do gradiente
                    optimizer.step() #alteras os valores de pesos e bias

                elif tipo == "Non stocastic":
                    for i in range(len(x_true)):
                        # foward pass
                        y_pred = self.foward(x_true[i]).squeeze() #calcula para o sample i
                        loss = loss_func(y_pred, y_true[i]) 
                        train_loss_sum += loss.item() # tensor.item() retorna o valor nummerico do tensor caso seja 1x1

                        #backward
                        optimizer.zero_grad() #zera os valores .grad calculados
                        loss.backward() # cálculo do gradiente
                        optimizer.step() #alteras os valores de pesos e bias

                elif tipo == "Stocastic":
                    raise NameError("tipo não implementado")
                
                elif tipo == "Minibatch":
                    raise NameError("tipo não implementado")
                
                train_loss_sum_vec.append(train_loss_sum)
        else:
            # talvez definir para com lr fixo aqui?
            raise NameError("otimizador não definido")
        
        return train_loss_sum_vec
    
    def treino_validacao(self, x_train, y_train, x_val, y_val, learning_rate=0.01, number_epochs=4000,
                         loss_func=nn.MSELoss(), tipo = "Batch", optimize = "Adam", zerar_seed=False,
                          Mini_batch_size = None, seed = 0):
        train_loss_sum_vec = []
        val_loss_sum_vec = []
        val_loss_min, val_loss_max = 10**100, 0

        if zerar_seed:
            t.manual_seed(seed)

        if optimize == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr = learning_rate)

            for epoch in range(number_epochs):
                self.train() # permite ao programa treinar
                train_loss_sum, val_loss_sum = 0, 0

                if tipo == "Batch":
                    # foward pass de treinamento
                    y_pred = self.foward(x_train).squeeze()

                    #determinação da perda
                    loss = loss_func(y_pred, y_train.squeeze())
                    train_loss_sum += loss.item()

                    #backward automatic derivation
                    optimizer.zero_grad() #zera os valores .grad calculados
                    loss.backward() # cálculo do gradiente
                    optimizer.step() #alteras os valores de pesos e bias

                    #preparando o modelo para validação
                    self.eval()
                    with t.no_grad():
                        # foward pass de validação
                        y_pred = self.foward(x_val).squeeze()

                        #determinação da perda
                        loss = loss_func(y_pred, y_val.squeeze())
                        val_loss_sum += loss.item()

                elif tipo == "Non stocastic":
                    raise NameError("tipo não implementado")

                elif tipo == "Stocastic":
                    raise NameError("tipo não implementado")
                
                elif tipo == "Minibatch":
                    raise NameError("tipo não implementado")
                
                train_loss_sum_vec.append(train_loss_sum)
                val_loss_sum_vec.append(val_loss_sum)
                if val_loss_min > val_loss_sum:
                    val_loss_min = val_loss_sum
                    savestate = copy.deepcopy(self.state_dict())

                if val_loss_max < val_loss_sum:
                    val_loss_max = val_loss_sum
        else:
            # talvez definir para com lr fixo aqui?
            raise NameError("otimizador não definido")
        
        return train_loss_sum_vec, val_loss_sum_vec, val_loss_min, val_loss_max, savestate
    
class neural_net_interno_1_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        return x
    
class neural_net_interno_2_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        x = self.func(x)
        x = self.fc3(x)
        return x
    
class neural_net_interno_3_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])
        self.fc4 = nn.Linear(sizes[3], sizes[4])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        x = self.func(x)
        x = self.fc3(x)
        x = self.func(x)
        x = self.fc4(x)
        return x
    
class neural_net_interno_4_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])
        self.fc4 = nn.Linear(sizes[3], sizes[4])
        self.fc5 = nn.Linear(sizes[4], sizes[5])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        x = self.func(x)
        x = self.fc3(x)
        x = self.func(x)
        x = self.fc4(x)
        x = self.func(x)
        x = self.fc5(x)
        return x

class neural_net_interno_5_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])
        self.fc4 = nn.Linear(sizes[3], sizes[4])
        self.fc5 = nn.Linear(sizes[4], sizes[5])
        self.fc6 = nn.Linear(sizes[5], sizes[6])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        x = self.func(x)
        x = self.fc3(x)
        x = self.func(x)
        x = self.fc4(x)
        x = self.func(x)
        x = self.fc5(x)
        x = self.func(x)
        x = self.fc6(x)
        return x
    
class neural_net_interno_6_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "ReLU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super().__init__(activation_func)
        
        if zerar_seed:
            t.manual_seed(0)
         
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])
        self.fc4 = nn.Linear(sizes[3], sizes[4])
        self.fc5 = nn.Linear(sizes[4], sizes[5])
        self.fc6 = nn.Linear(sizes[5], sizes[6])
        self.fc7 = nn.Linear(sizes[6], sizes[7])

    def foward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        x = self.func(x)
        x = self.fc3(x)
        x = self.func(x)
        x = self.fc4(x)
        x = self.func(x)
        x = self.fc5(x)
        x = self.func(x)
        x = self.fc6(x)
        x = self.func(x)
        x = self.fc7(x)
        return x
    
class plt_plot_general():
    def __init__(self, figsize = (8,24)):
        self.fig = plt.figure(figsize=figsize)
    
    def plot_3d_surface(self, model, xa, xb, func, pos = [3, 1, 1], label_true = 'função a estimar', label_pred = "resultado da rede"):
        self.model = model
        self.xa = xa
        self.xb = xb

        #criando malha
        grid_step = 50
        xs = t.linspace(self.xa, self.xb, steps=grid_step)
        ys = t.linspace(self.xa, self.xb, steps=grid_step)
        xg,yg = t.meshgrid(xs, ys, indexing='xy')
        s = xs.size(0)
        xyg = t.zeros([s,s,2])

        xyg[:,:,0] = xg
        xyg[:,:,1] = yg

        #fazendo predição da rede
        y_pred = self.model.foward(xyg)
        y_pred = t.squeeze(y_pred)

        #definindo resultados exatos
        y_true = func(xg, yg)

        #plotando
        ax = self.fig.add_subplot(pos[0], pos[1], pos[2], projection='3d')
        ax.plot_surface(xg, yg, y_pred.detach().numpy(), cmap=cm.Blues, alpha=0.6,label=label_pred, linewidth=1, edgecolor='blue')
        ax.plot_surface(xg, yg, y_true, cmap=cm.Reds, alpha=0.6,label=label_true, linewidth=1, edgecolor='red')
        ax.set_xlabel('X1-axis')
        ax.set_ylabel('X2-axis')
        ax.set_zlabel('Y-axis')
        ax.legend(loc = 'upper right')

    def plot_erros(self, train_loss_sum_vec, val_loss_sum_vec, pos = [3, 1, 2]):
        ax2 = self.fig.add_subplot(pos[0], pos[1], pos[2])
        
        ax2.plot(range(len(train_loss_sum_vec)), train_loss_sum_vec, label = "erro de treinamento")
        ax2.plot(range(len(val_loss_sum_vec)), val_loss_sum_vec, label = "erro de validação")
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.grid(True)
        ax2.legend(loc = 'upper left')

    def plot_training_points(self, train_points, val_points, pos = [3, 1, 3]):
        ax3 = self.fig.add_subplot(pos[0], pos[1], pos[2])

        ax3.scatter(train_points[:,0], train_points[:,1], label = "pontos de treinamento")
        ax3.scatter(val_points[:,0], val_points[:,1], label = "pontos de validação")
        ax3.grid(True)
        ax3.legend()

    def show(self):
        # Ajuste o layout para evitar sobreposição
        self.fig.tight_layout()
        #mostrandoo a figura
        plt.show()

    def save_fig(self, path):
        # Ajuste o layout para evitar sobreposição
        self.fig.tight_layout()
        #salvamento
        self.fig.savefig(path)

    def close(self):
        plt.close(self.fig)

def plotagem_1d_modelo(xf, func, y_pred_ext, x_train, y_train, x_val, y_val, loss_train, loss_val, func_label, path, number_epochs=4000):
    '''
    xf: intervalo onde plotar a função
    func: função exata
    y_pred_ext: valores preditos pelo modelo para o intervalo de extrapolação
    x_train: valores x de trainamento
    y_train: valores y de treinamento
    x_val: valores x de validação
    y_val: valores y de validação
    loss: vetor com as perdas
    func_label: texto a associar com função exata
    path: caminho e nome do arquivo para salvar
    '''
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (7, 10))
    ax.plot(xf, func(xf), label = func_label)
    ax.plot(xf, y_pred_ext.detach().numpy(), label = "dados previstos")
    ax.scatter(x_train, y_train, label = "valores de treinamento")
    ax.scatter(x_val, y_val, label = "valores de validação")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax2.set_title("Evolução da perda")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.plot(range(number_epochs), loss_train, label = "Perda de treinamento")
    ax2.plot(range(number_epochs), loss_val, label = "Perda de validação")
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)