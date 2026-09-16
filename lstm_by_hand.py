import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class lstm_by_hand(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        # create and init weights and biases
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        # init input params
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # init output params
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # init forget params
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # init candidate params
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # output layer: hidden state -> single prediction
        self.fc = nn.Linear(hidden_sz, 1)

        self.init_params()
    
    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for param in self.parameters():
            param.data.uniform_(-stdv, stdv)
    
    
    def lstm_unit(self, x_t):
        # for LSTM math: Forget gate, input gate, output gate

        self.i_t = torch.sigmoid(self.W_i.T @ x_t + self.U_i.T @ self.h_t + self.b_i)
        self.f_t = torch.sigmoid(self.W_f.T @ x_t + self.U_f.T @ self.h_t + self.b_f)
        self.o_t = torch.sigmoid(self.W_o.T @ x_t + self.U_o.T @ self.h_t + self.b_o)
        self.can_t = torch.tanh(self.W_c.T @ x_t + self.U_c.T @ self.h_t + self.b_c)

        self.c_t = self.f_t * self.c_t + self.i_t * self.can_t
        self.h_t = self.o_t * torch.tanh(self.c_t)
    
    def forward(self, input):
        # forward pass 
        self.h_t = torch.zeros(self.hidden_sz) # h_0
        self.c_t = torch.zeros(self.hidden_sz) # c_0
        outputs = []
        for x_t in input:
            self.lstm_unit(x_t)
            outputs.append(self.h_t)

        # final hidden state -> single prediction
        return self.fc(self.h_t)

    def loss_function(self, prediction, target):
        return torch.mean((prediction - target) ** 2)

    def optimizer_function(self, learning_rate = 0.001):
        # adam optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )
        

    def training_step(self, X, y, optimizer):
        # things to do within a training step or epoch i.e calcualte loss
        
        predictions = self.forward(X)
        loss = self.loss_function(y, predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


if __name__ == "__main__":
    model = lstm_by_hand(input_sz=3, hidden_sz=10)

    optimizer = model.optimizer_function(learning_rate=0.001)
    X_train = []
    y_train = []
    for _ in range(1000):

        sequence = torch.randn(5, 3) * 5

        target = sequence.sum()

        X_train.append(sequence)
        y_train.append(target)

    # -----------------------------------
    # Training
    # -----------------------------------

    for epoch in range(100):

        total_loss = 0

        for X, y in zip(X_train, y_train):

            loss = model.training_step(
                X,
                y,
                optimizer
            )

            total_loss += loss

        avg_loss = total_loss / len(X_train)
        if epoch % 10 == 0:
            print("epoch:", epoch, "loss", avg_loss)

            test_X = torch.Tensor([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0]
            ])

            prediction = model(test_X)
            print("prediction at epoch: ", prediction.item())

    print("FINAL PREDICTION:", prediction.item())
    print("Actual:", 45.0)