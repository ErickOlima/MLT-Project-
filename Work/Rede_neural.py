import torch as t
import torch.nn as nn
import torch.optim as optim

class neural_net_interno(nn.Module):
    def __init__(self, activation_func = "RelU"):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super(neural_net_interno, self).__init__()

        if activation_func == "RelU":
            self.func = nn.ReLU()
        elif activation_func == "sigmoid":
            self.func = nn.Sigmoid()
        elif activation_func == "tanh":
            self.func = nn.Tanh()
        else: 
            raise NameError("função de ativação não definida")
    
    def treino(self, x_true, y_true, learning_rate=0.01, number_epochs=4000, loss_func=nn.MSELoss(),
                  tipo = "Batch", optimize = "Adam", zerar_seed=False, Mini_batch_size = None):
        train_loss_sum_vec = []

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
        
class neural_net_interno_1_hidden(neural_net_interno):
    def __init__(self, sizes:list, activation_func = "RelU", zerar_seed=False):
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
    def __init__(self, sizes:list, activation_func = "RelU", zerar_seed=False):
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
    def __init__(self, sizes:list, activation_func = "RelU", zerar_seed=False):
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
    def __init__(self, sizes:list, activation_func = "RelU", zerar_seed=False):
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