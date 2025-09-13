import torch as t
import torch.nn as nn
import torch.optim as optim

class neural_net_interno(nn.Module):
    def __init__(self, sizes:list, activation_func = "RelU", zerar_seed=False):
        '''
        sizes : [input_size, hidden_size1, hidden_size2, ..., output_size]
        activation_func : função de ativação aplicada interiormente
        '''
        super(neural_net_interno, self).__init__()
        if zerar_seed:
            t.manual_seed(0)
            
        fc = []
        for i in range(len(sizes)-1):
            fc.append(nn.Linear(sizes[i], sizes[i+1]))
        self.fc = fc

        if activation_func == "RelU":
            self.func = nn.ReLU()
        elif activation_func == "sigmoid":
            self.func = nn.Sigmoid()
        elif activation_func == "tanh":
            self.func = nn.Tanh()
        else: 
            raise NameError("função de ativação não definida")

    def foward(self, x):
        for i in range(len(self.fc)-1):
            x = self.fc[i](x)
            x = self.func(x)
        x = self.fc[-1](x)
        return x
    
    def training(self, x_true, y_true, learning_rate=0.01, number_epochs=4000, loss_func=nn.MSELoss(),
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
                    loss = loss_func(y_pred, y_true)
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
        
    