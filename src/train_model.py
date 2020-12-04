
"""
Need to download these packages on cloud: 
# conda install pytorch torchvision -c pytorch
# pip install -U gensim
#>> python 
#>> import nltk
#>> nltk.download('punkt')
"""

# Citation:
# https://pytorch.org/docs/master/nn.html,
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
# https://pytorch.org/docs/stable/nn.html?highlight=pack_padd#torch.nn.utils.rnn.pack_padded_sequence
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/

# Imports
from typing import List
import nltk
import numpy as np
import pandas as pd
import pickle
import io
from numpy.core._multiarray_umath import ndarray
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
# Experiment with another embedding file
import utils_new

# Global definitions - data
DATA_FN = "./data/crowdflower_data.csv"
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    prev_loss = np.Infinity
    stop = False
    epoch = 0
    trained_model = model  # to hold best model
    while not stop:
        hist_train_loss = []
        # Set network into train set
        model.train()
        for batch_x, batch_y in train_generator:
            # Reset optimizer
            optimizer.zero_grad()
            # Predict outputs
            outputs = model(batch_x)
            # Calculate the loss
            loss = loss_fn(outputs, batch_y)
            hist_train_loss.append(loss.cpu().detach().numpy())
            # Backward and update step
            loss.backward()
            optimizer.step()

        epoch += 1

        # Set network into development set
        model.eval()
        val_gold = []
        val_pred = []
        loss = 0

        with torch.no_grad(): # set not gradient
            optimizer.zero_grad()
            for batch_x, batch_y in dev_generator:
                outputs = model(batch_x)
                # Add predictions and gold labels
                val_gold.extend(batch_y.cpu().detach().numpy())
                val_pred.extend(outputs.argmax(1).cpu().detach().numpy())
                loss += loss_fn(outputs.double(), batch_y.long()).data

            f1 = f1_score(val_gold, val_pred, average='macro')
            print('Epoch: ' + str(epoch) + ', Total dev Loss: ' + str(loss.numpy()) + ', Total dev f-1: ' + str(f1))

            if loss < prev_loss:
                prev_loss = loss
                trained_model = model
            else:
                stop = True

    return trained_model


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)
            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main():
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    print("build model")
    # If GPU available, set device to GPU.
    # Otherwise use CPU.
    if USE_CUDA:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Tried hidden dimension: 64, 128, 256, 521, 1024
    model = models.DenseNetwork(embeddings, hidden_dim=256).to(device)
    # print(model)
    # model = torch.load('./dense.pth')
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=0.001)#.cuda()
    print("training dense...")
    trained_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator)
    torch.save(trained_model, './dense.pth')
    test_model(trained_model, loss_fn, test_generator)

    # Tried hidden dimension: 32, 64, 128
    # model_rnn = torch.load('./recurrent.pth')
    # model_rnn.eval()
    model_rnn = models.RecurrentNetwork(embeddings, input_size=100, hidden_dim=64, n_layers=2).to(device)
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
    print("training rnn...")
    trained_model = train_model(model_rnn, loss_fn, optimizer_rnn, train_generator, dev_generator)
    torch.save(trained_model, './recurrent.pth')
    test_model(trained_model, loss_fn, test_generator)


    # Uncomment it to load new embeddings
    ''' 
    # Experiment with another embeddings
    print("Preprocessing all data from scratch.... EXPERIMENT WITH NEW EMEBDDINGs")
    train, dev, test = utils_new.get_data(DATA_FN)
    # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
    train_generator, dev_generator, test_generator, embeddings, train_data = utils_new.vectorize_data(train, dev, test,
                                                                                                  BATCH_SIZE,
                                                                                                 EMBEDDING_DIM2)
    print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
          "False to load them from file....")
    with open('300_42B_temp.pkl', "wb+") as f:
        pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f) 
    '''

    # Uncomment it to do experiment DNN and RNN.
    '''
    try:
        with open('300_42B_temp.pkl', "rb") as f:
            print("Loading DataLoaders and embeddings from file....")
            train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    EMBEDDING_DIM2 = 300
    # Experiment with cnn
    model_cnn = models.ExperimentalNetwork(embeddings, embed_dim=EMBEDDING_DIM2).to(device)
    optimizer_rnn = optim.Adam(model_cnn.parameters(), lr=0.001)
    print("training experimental CNN...")
    trained_model = train_model(model_cnn, loss_fn, optimizer_rnn, train_generator, dev_generator)
    test_model(trained_model, loss_fn, test_generator)

    # Experiment with bilstm
    model_bilstm = models.ExperimentalRNN(embeddings, embed_dim=EMBEDDING_DIM2, hidden_size=64).to(device)
    optimizer_rnn = optim.Adam(model_bilstm.parameters(), lr=0.001)
    print("training experimental biLSTM...")
    trained_model = train_model(model_bilstm, loss_fn, optimizer_rnn, train_generator, dev_generator)
    test_model(trained_model, loss_fn, test_generator)
    '''


if __name__ == '__main__':
    main()
