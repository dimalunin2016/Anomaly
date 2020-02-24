import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from sklearn.preprocessing import StandardScaler
import gc
import scipy.stats as sps
import random
import numpy as np
import pandas as pd
from AnomalyModelABS import AnomalyModel


def generate_batch_data(x, y, batch_size):
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch


def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x)
    y_arr = np.array(y)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var


class SimpleLSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, answers, train_size, future_size):
        outputs = []
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(answers.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(answers.size(0), self.hidden_size, dtype=torch.float32)

        for i in range(train_size):
            current_ans = answers[:, [i]]
            h_t, c_t = self.lstm(current_ans[:, 0, :], (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(train_size, train_size + future_size):
            if random.random() > 0.5:
                output = answers[:, [i]]  # teacher forcing
                output = output[:, 0, :]
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        return outputs


def train_model_part(model, train_data, train_size, future_size,
                     loss_func, optimizer,
                     num_epochs = 150, 
                     batch_size=200):
    train_window = train_size + future_size
    x_train, y_train = transform_data(train_data, train_window)
    for epoch in range(num_epochs):
        train_loss = 0
        for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):
            y_pred = model(x_batch, train_size = train_size, future_size = future_size)
            optimizer.zero_grad()
            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= batch_size
    return model


class SimpleLSTMAnomalyModel(AnomalyModel):
    def __init__(self, in_out_size, learning_rate=0.01, 
                 hidden_lstm_size=21, 
                 last_scores_size=8000,
                 small_window_size=10,
                 train_size=90,
                 train_data_size=1000,
                 num_epochs=500,
                 batch_size=300):

        super().__init__()
        self.lstm = SimpleLSTModel(in_out_size, hidden_lstm_size, in_out_size)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate)
        self.last_scores_size = last_scores_size
        self.small_window_size = small_window_size
        self.future_size = small_window_size
        self.train_size = train_size
        self.last_scores = []
        self.small_window_scores = []
        self.trained = False
        self.start_predicting = False
        self.train_data_size = train_data_size
        self.train_data = []
        self.curr_sample = []
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    
    def train_model(self):
        self.lstm = train_model_part(self.lstm, 
                                self.train_data, self.train_size, 
                                self.future_size,
                                self.loss_func, self.optimizer, 
                                num_epochs=self.num_epochs, 
                                batch_size=self.batch_size)
        self.train_data = []
        
    def predict_anomaly_proba(self, point)->float:
        if not self.trained:
            self.train_data.append(point.tolist())
            if len(self.train_data) == self.train_data_size:
                self.train_model()
                self.trained = True
            return 0
        else:
            self.curr_sample.append(point.tolist())
            if len(self.curr_sample) > self.train_size + self.future_size:
                train_sample = self.curr_sample[:-1]
                self.curr_sample = self.curr_sample[1:]
 
                y_sample = Variable(torch.from_numpy(
                    np.array([self.curr_sample])).float())
                train_sample = Variable(torch.from_numpy(
                    np.array([train_sample])).float())

                y_pred = self.lstm(train_sample, train_size = self.train_size,
                              future_size = self.future_size)
                self.optimizer.zero_grad()
                loss = self.loss_func(y_pred, y_sample)
                loss.backward()
                self.optimizer.step()
                self.last_scores.append(loss.item())
                self.small_window_scores.append(loss.item())
                self.start_predicting = True
        if len(self.small_window_scores) > self.small_window_size:
            self.small_window_scores = self.small_window_scores[1:]
        if len(self.last_scores) > self.last_scores_size:
            self.last_scores = self.last_scores[1:]
        if self.start_predicting:
            big_mu = np.mean(self.last_scores)
            small_mu = np.mean(self.small_window_scores)
            sigma = np.std(self.last_scores)
            pvalue = 2 * sps.norm(0, 1).cdf(abs(small_mu - big_mu) / sigma) - 1
            return pvalue
        else:
            return 0

