import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
from AnomalyModelABS import AnomalyModel
import rrcf


class RobustRandomCutForest(AnomalyModel):
    """RobustRandomCutForest model"""

    def __init__(self, number_of_trees=40, train_size=500,
                 queue_size=500,
                 last_scores_size=8000,
                 small_window_size=10,):
        super().__init__()
        self.forest = [rrcf.RCTree() for _ in range(number_of_trees)]
        self.train_size = train_size
        self.current_index = 0
        self.queue_size = queue_size
        self.anomaly_scores_queue = []
        self.points_to_add_in_future = []
        self.last_scores_size = last_scores_size
        self.small_window_size = small_window_size
        self.queue = []
        self.anomaly_scores_queue = []

    def __add_point_in_forest(self, point, index):
        for tree in self.forest:
            tree.insert_point(point, index=index)

    def __remove_point(self, index):
        for tree in self.forest:
            tree.forget_point(index)

    def __calculate_anomaly_score(self, point, index):
        score = 0

        for tree in self.forest:
            tree.insert_point(point, index=index)
            new_codisp = tree.codisp(index)
            score += new_codisp / len(self.forest)
        return score

    def predict_anomaly_proba(self, point)->float:
        self.current_index += 1
        self.queue.append(self.current_index)
        if self.current_index <= self.train_size:
            self.__add_point_in_forest(point, self.current_index)
            return 0

        score = self.__calculate_anomaly_score(point, self.current_index)
        self.anomaly_scores_queue.append(score)
        
        if len(self.anomaly_scores_queue) > self.last_scores_size:
            self.anomaly_scores_queue = self.anomaly_scores_queue[1:]
        if len(self.queue) > self.queue_size:
            self.__remove_point(self.queue[0])
            self.queue = self.queue[1:]
        
        if len(self.anomaly_scores_queue) < self.small_window_size:
            return 0
        current_score = self.anomaly_scores_queue[-1]
        big_mu = np.mean(self.anomaly_scores_queue)
        small_mu = np.mean(self.anomaly_scores_queue[-self.small_window_size: ])
        sigma = np.std(self.anomaly_scores_queue)
        pvalue = 2 * sps.norm(0, 1).cdf(abs(small_mu - big_mu) / sigma) - 1
        return pvalue
