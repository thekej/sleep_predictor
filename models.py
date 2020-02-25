import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

import math

class SleepPredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=3,
                 num_layers=1, dropout_p=0.0, w_norm=False):
        super(SleepPredictionMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.w_norm = w_norm
        layers = []
        
        for i in range(num_layers):
            idim = hidden_size
            odim = hidden_size
            if i == 0:
                idim = input_size
            if i == num_layers-1:
                odim = num_classes
            fc = nn.Linear(idim, odim)
            fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
            fc.bias.data.fill_(0)
            
            if w_norm:
                fc = weight_norm(fc, dim=None)
            layers.append(fc)

            if i != num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(odim)),
                layers.append(nn.Dropout(p=dropout_p))
                
        self.layers = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        out = self.layers(x)
        return out
    
        
class SleepPredictionCNN(nn.Module):
    def __init__(self, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=125, 
                 num_classes=3, dropout_p = 0.0):
        super(SleepPredictionCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size +1
            conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, 
                               kernel_size = kernel_size, 
                               stride = 1)
            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            )

            conv_blocks.append(component)
        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x, h0=None):       # x: (batch, sentence_len)
        #x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1,2)    # needs to convert x to (batch, embedding_dim, sentence_len)
        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return F.softmax(self.fc(out), dim=1)
    

class SleepPredictionSeq(nn.Module):
    def __init__(self, hidden_size, input_dropout_p=0, 
                 dropout_p=0, n_layers=1, embed_size=125,
                 bidirectional=False, rnn_cell='lstm'):
        super(SleepPredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.linear = nn.Linear(11, embed_size)
        self.rnn_cell = getattr(nn, rnn_cell.upper())
        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout_p)

    def forward(self, x, h0=None):
        if h0 is not None:
            h0 = self.linear(h0)
            if self.rnn_cell is nn.LSTM:
                h0 = (h0, h0)
        embedded = self.input_dropout(x)
        output, hidden = self.rnn(embedded, h0)
        return output, hidden
