# -*- coding: utf-8 -*-
"""

@author: Filipe

Here you will find a pytorch implementation from scratch of RNN, LSTM and GRU. 
I tried to divide the steps into functions to be easir to understand the whole process
and tried to do the way pytorch does internally to be easier to use pytorch later;

In this notebook: https://colab.research.google.com/drive/1hrsAgirFaZQ-z0qui9YXRiDnuUmt5eRX#scrollTo=VZiWEXwmXtkZ
You can see it working in practice for sentence generation.
"""

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size  = output_size
        self.input_size = input_size

        self.encode = nn.Linear(input_size+hidden_size, hidden_size)
        self.decode_latent = nn.Linear(hidden_size, output_size)
    
    def decode(self,h):
        return self.decode_latent(h)

    def predict(self,outs):
        return torch.argmax(torch.softmax(outs, dim=0), dim=1)
    
    def predict_proba(self,outs):
        return torch.softmax(outs, dim=0)
        
    def one_step(self, x, h):
        x_h = torch.cat((x, h), axis=1)
        return torch.sigmoid(self.encode(x_h))

    def forward(self, x):

        # Initialize hidden state for first iteration
        h = torch.zeros((x.shape[0], self.hidden_size)).to(x.device)
        hidden_states = []

        # Iterative forward
        for i in range(x.shape[1]):
            h = self.one_step(x[:,i], h)
            hidden_states.append(h)
            
        return torch.stack(hidden_states)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        #W corresponds to Wf, Wi, Wo and Wc, and the same for U.
        
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=True)


    def forward(self, x, h, c):
        
        gates = self.W(x) + self.U(h)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = self.sigmoid(input_gate)
        f_t = self.sigmoid(forget_gate)
        g_t = self.tanh(cell_gate)
        o_t = self.sigmoid(output_gate)

        cy = c * f_t + i_t * g_t

        hy = o_t * self.tanh(cy)

        return hy, cy
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size


        self.rnnCell = LSTMCell(self.input_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
  
    def one_step(self, x, h, c):
        h, c = self.rnnCell(x, h, c)
        return h, c

    def decode(self,h):
        return self.fc(h)

    def forward(self, x):

        h = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        c = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        
        hidden_states = []
        c_states = []
        
        #for all words/time stemps
        for t in range(x.shape[1]):
            #If you want more than 1 layer
            for layer in range(self.num_layers):
                h, c = self.one_step(x[:,t], h, c)
                hidden_states.append(h)
                c_states.append(c)
        
        return torch.stack((torch.stack(hidden_states),torch.stack(c_states)))


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        #W corresponds to Wz, Wr and Wh, and the same for U.
        
        self.W = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.U = nn.Linear(hidden_size, 3 * hidden_size, bias=True)


    def forward(self, x, h):
        
        x_t = self.W(x)
        h_t = self.U(h)

        # As I have Wz, Wr and Wh on W and the same for U, I need to chunk them
        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)
        
        #Setting the gates
        reset_gate = self.sigmoid(x_reset + h_reset)
        update_gate = self.sigmoid(x_upd + h_upd)
        new_gate = self.tanh(x_new + (reset_gate * h_new))

        #Next state
        hy = update_gate * h + (1 - update_gate) * new_gate

        return hy

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size


        self.rnnCell = GRUCell(self.input_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def one_step(self, x, h):
        h = self.rnnCell(x, h)
        return h

    def decode(self,h):
        return self.fc(h)

    def forward(self, x):

        h = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        
        hidden_states = []
        
        #for all words/time stemps
        for t in range(x.shape[1]):
            #If you want more than 1 layer
            for layer in range(self.num_layers):
                h = self.one_step(x[:,t], h)
                hidden_states.append(h)
        
        return torch.stack(hidden_states)