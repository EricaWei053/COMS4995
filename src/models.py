
"""
Need to download these packages on cloud: 
# conda install pytorch torchvision -c pytorch
# pip install -U gensim

#>> python 
#>> import nltk
#>> nltk.download('punkt')
"""

# Citation:
# https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
# https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/


import torch
import torch.nn as nn
#import gensim
import numpy as np
import torch.nn.functional as F
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
cd = os.getcwd()
import torch.nn.utils.rnn as rnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 20000


class DenseNetwork(nn.Module):
    def __init__(self, embeddings, hidden_dim):
        super(DenseNetwork, self).__init__()

        
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.fc1 = nn.Linear(in_features=100, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=4)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
       
        x = self.embedding(x)
        x = x.sum(1)
        # First layer
        x = self.fc1(x)
        #x = self.tanh(x)
        #x = self.dropout(x)
        x = self.sigmoid(x)
        # Second layer
        x = self.fc2(x)

        return x

class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, input_size, hidden_dim, n_layers):
        super(RecurrentNetwork, self).__init__()
        
        # Pre-trained embedding layer
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.dropout = 0.1

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=self.dropout)
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=self.dropout)
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=self.dropout)
        # Fully connected layer
        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=4)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
       
        # Embedding layer
        embeds = self.embedding(x)
        # Pack padded sequence
        input_lengths = [len(torch.nonzero(seq)) for seq in x]
        packed = pack_padded_sequence(embeds, input_lengths, batch_first=True, enforce_sorted=False)

        # Initializing hidden state for first input using method defined below
        
        rnn_out, hidden = self.gru(packed)

     
        # Unpacked it
        output, _ = pad_packed_sequence(rnn_out)
        # Project the final hidden state to a dense layer
        out = self.hidden2tag(hidden[-1])
        return out


class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings, embed_dim):
        super(ExperimentalNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        filter_sizes = [1, 2, 3, 5]
        num_filters = 256
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.conv_list = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_dim)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 4)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(cv(x)).squeeze(3) for cv in self.conv_list]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        #print(x.shape)
        x = self.dropout(x)
        out = self.fc1(x)
        return out


class ExperimentalRNN(nn.Module):
    def __init__(self, embeddings, embed_dim, hidden_size):
        super(ExperimentalRNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, 4)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = torch.squeeze(torch.unsqueeze(embeds, 0))
        lstm_o, _ = self.lstm(embeds)
        avg_pool = torch.mean(lstm_o, 1)
        max_pool, _ = torch.max(lstm_o, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.relu(self.linear(out))
        out = self.dropout(out)
        out = self.out(out)
        return out


