import torch as t
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


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

class plot_error():
    def __init__(self, train_loss_sum_vec, path):
        self.train_loss_sum_vec = train_loss_sum_vec
        self.path = path
        
        plt.plot(range(len(train_loss_sum_vec)), train_loss_sum_vec)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True)
        plt.show()


class plt_3d_surface():
    def __init__(self, model, xa, xb):
        self.model = model
        self.xa = xa
        self.xb = xb

    def plot(self):
        grid_step = 50
        xs = t.linspace(self.xa, self.xb, steps=grid_step)
        ys = t.linspace(self.xa, self.xb, steps=grid_step)
        xg,yg = t.meshgrid(xs, ys, indexing='xy')
        s = xs.size(0)
        xyg = t.zeros([s,s,2])

        xyg[:,:,0] = xg
        xyg[:,:,1] = yg

        y_pred = self.model.forward(xyg)
        y_pred = t.squeeze(y_pred)

        y_true = self.func(xg, yg)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xg, yg, y_pred.detach().numpy(), vmin=0. * 2, cmap=cm.Blues, alpha=0.5,label='y pred (trained net)')
        ax.plot_surface(xg, yg, y_true, vmin=0.0 * 2, cmap=cm.Reds, alpha=0.5,label='y true')
        ax.set_xlabel('X1-axis')
        ax.set_ylabel('X2-axis')
        ax.set_zlabel('Y-axis')
        ax.legend()
        plt.show()