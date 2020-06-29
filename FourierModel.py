import gc
import scipy.stats as sps
import random
import numpy as np
import pandas as pd
from AnomalyModelABS import AnomalyModel


class FourierAnomalyModel(AnomalyModel):
    def __init__(self, dimension,
                 train_window_size=1000,
                 predict_window_size=100,
                 last_scores_size=8000,
                 small_window_size=10,
                 n_harm=40):
        super().__init__()
        self.train_window_size = train_window_size
        self.predict_window_size = predict_window_size
        self.n_harm = n_harm
        self.last_scores_size = last_scores_size
        self.small_window_size = small_window_size
        self.last_scores = [[] for _ in range(dimension)]
        self.queueries = [[] for _ in range(dimension)]

    @staticmethod
    def __fourierExtrapolation(x, n_predict, n_harm):
        n = x.size
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)         # find linear trend in x
        x_notrend = x - p[0] * t        # detrended x
        x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
        f = np.fft.fftfreq(n)              # frequencies
        indexes = np.arange(0, n)
        sorted(indexes,key = lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    def predict_anomaly_proba(self, point):
        for ind, val in enumerate(point.tolist()):
            self.queueries[ind].append(val)

        if len(self.queueries[0]) == self.train_window_size + self.predict_window_size:
            probs = []
            for ind in range(len(self.queueries)):
                query = self.queueries[ind]
                train = query[:self.train_window_size]
                y_true = query[self.train_window_size:]
                predictions =\
                FourierAnomalyModel.__fourierExtrapolation(np.array(train),
                                                                self.predict_window_size,
                                                                self.n_harm)
                predictions = predictions[self.train_window_size:]
                diffs = np.mean(np.abs(predictions - y_true) ** 2)
                self.last_scores[ind].append(diffs)
                if len(self.last_scores[ind]) <= self.small_window_size:
                    probs.append(0)
                else:
                    big_mu = np.mean(self.last_scores[ind])
                    small_mu = np.mean(self.last_scores[ind][-self.small_window_size:])
                    sigma = np.std(self.last_scores[ind]) + 10 **-5
                    pvalue = 2 * sps.norm(0, 1).cdf(abs(small_mu - big_mu) / sigma) - 1
                    probs.append(pvalue)
                    if len(self.last_scores[ind]) == self.last_scores_size:
                        self.last_scores[ind] = self.last_scores[ind][1:]
                self.queueries[ind] = self.queueries[ind][1:]
            return max(probs)
        else:
            return 0
