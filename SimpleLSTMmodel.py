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
            h_t, c_t = self.lstm(current_ans, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(train_size, train_size + future_size):
            if random.random() > 0.5:
                output = answers[:, [i]]  # teacher forcing
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
                 train_size=90):

        super().__init__()
        self.lstms = [SimpleLSTModel(1, hidden_lstm_size, 1) for _ in range(in_out_size)]
        self.loss_func = torch.nn.MSELoss()
        self.optimizers = [
            torch.optim.Adam(self.lstms[ind].parameters(), lr=learning_rate) 
            for ind in range(in_out_size)]
        self.last_scores_size = last_scores_size
        self.small_window_size = small_window_size
        self.future_size = small_window_size
        self.train_size = train_size
        self.last_scores = [[] for _ in range(in_out_size)]
        self.train_data = []
        self.curr_samples = [[] for _ in range(in_out_size)]
        
    def predict_anomaly_proba(self, point)->float:
        
        curr_vals = point.tolist()
        pvals = []
        for ind, val in enumerate(curr_vals):
            self.curr_samples[ind].append(val)
            if len(self.curr_samples[ind]) > self.train_size + self.small_window_size:
                size = self.train_size + self.small_window_size
                curr_sample = np.array([self.curr_samples[ind][1:]]).reshape(-1, size)
                train_sample = np.array([self.curr_samples[ind][:-1]]).reshape(-1, size)
                y_sample = Variable(torch.from_numpy(
                    np.array(curr_sample).reshape(curr_sample.shape[0], 
                                                  curr_sample.shape[1], 1)).float())
                train_sample = Variable(torch.from_numpy(
                    np.array(train_sample)).float())
                self.optimizers[ind].zero_grad()
                y_pred = self.lstms[ind](train_sample, train_size = self.train_size,
                                          future_size = self.future_size)
                loss = self.loss_func(y_pred, y_sample)
                loss.backward()
                self.optimizers[ind].step()
                self.last_scores[ind].append(loss.item())
                self.curr_samples[ind] = self.curr_samples[ind][1:]
            if len(self.last_scores[ind]) > self.last_scores_size:
                self.last_scores[ind] = self.last_scores[ind][1:]
            if len(self.last_scores[ind]) > self.small_window_size:
                big_mu = np.mean(self.last_scores[ind])
                small_mu = np.mean(self.last_scores[ind][-self.small_window_size:])
                sigma = np.std(self.last_scores[ind])
                pvalue = 2 * sps.norm(0, 1).cdf(abs(small_mu - big_mu) / sigma) - 1
                pvals.append(pvalue)
            else:
                pvals.append(0)
        return max(pvals)

