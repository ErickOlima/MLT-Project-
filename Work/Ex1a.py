import torch as t
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch.optim as optim
import Rede_neural as rn

# Resolvendo o problema 1D de forma mais simples 


def plotagem_1d_modelo(xf, func, y_pred, y_true, loss, func_label, path):
    '''
    xf: intervalo onde plotar a função
    func: função exata
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
    #plt.plot.show()

class Net_1_hidden_layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_func):
        
        super(Net_1_hidden_layer, self).__init__()
        
        t.manual_seed(0)
        
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, output_size) # Fully connected layer 2
        
        
        if activation_func=='ReLU':
          self.func = nn.ReLU()                         
        elif activation_func=='sigmoid':
          self.func = nn.Sigmoid()                      
        elif activation_func=='tanh':
          self.func = nn.Tanh()                         
        else:
          raise NameError('activation_func function ' + activation_func + ' not supported yet!')

    
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.func(out)
        out = self.fc2(out)
        return out


# modelo ReLU 
modelo_1 = { 
    'input_size': 1,
    'hidden_size': 20,
    'output_size': 1,
    'activation_func': 'ReLU'
}

modelo_2 = { 
    'input_size': 1,
    'hidden_size_1': 30,
    'hidden_size_2': 30,
    'output_size': 1,
    'activation_func': 'ReLU'
}

modelo_3 = {
    'input_size': 1,
    'hidden_size_1': 5,
    'hidden_size_2': 5,
    'hidden_size_3': 5,
    'output_size': 1,
    'activation_func': 'ReLU'
}

modelo_4 = { 
    'input_size': 1,
    'hidden_size_1': 10,
    'hidden_size_2': 10,
    'hidden_size_3': 10,
    'hidden_size_4': 10,
    'output_size': 1,
    'activation_func': 'ReLU'
}

#model = rn.neural_net_interno_1_hidden([input_size, hidden_size, output_size], activation_func=func)
model = Net_1_hidden_layer(*modelo_1.values())

#função real a ser aproximada
func = lambda x: x ** 3

#amostras de treinamento

samples_number = 20
ax, bx = -4, 4

#.manual_seed(0)  # controlar aleatoriedade

x_true = ax + (bx - ax) * t.rand(samples_number, 1)  # amostras de treinamento
y_true = func(x_true)  # valores reais correspondentes


for activation_func in ("ReLU", "sigmoid", "tanh"):


    # 1 hidden layer

    model = rn.neural_net_interno_1_hidden([1, modelo_1['hidden_size'], 1], activation_func) # uma camada interna de 20]
    loss = model.treino(x_true, y_true)

    xf = np.linspace(-6, 6, 12)
    y_pred = model.foward(t.tensor(np.array([xf]).T, dtype = t.float32))

    error_measure = np.sum((func(xf)-y_pred.detach().numpy())**2)
    
    plotagem_1d_modelo(xf, func, y_pred, y_true, loss, "y = x³",
                        f"Work\Ex1_graphs\{modelo_1['activation_func']}_hidden1_{modelo_1['hidden_size']}")










"""
samples_number = 20 
ax, bx = -4, 4
torch.manual_seed(0)

x_true = ax + (bx - ax) * torch.rand(samples_number, modelo_1['input_size']) 
y_true = x_true**3

learning_rate = 0.01
number_epochs = 4000

loss_function = nn.MSELoss() 
train_loss_sum_vec = [] 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(number_epochs):
  
  model.train() 
  
  y_pred = model(x_true) 
  
  loss = loss_function(y_pred,y_true)
  train_loss_sum = loss.item()
  
  
  optimizer.zero_grad() 
  loss.backward() 
  optimizer.step() 
  train_loss_sum_vec.append(train_loss_sum) 
  
y_pred = model.forward(x_true) 
plt.plot(x_true,y_true,"*",label='y true (training data)')
plt.plot(x_true,y_pred.detach().numpy(),"^",label='y pred (trained net)')
plt.legend()
plt.grid(True)
plt.show()
"""


