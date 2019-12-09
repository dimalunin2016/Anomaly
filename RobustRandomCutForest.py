import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AnomalyModelABS import AnomalyModel
import rrcf


class LDCDRobustRandomCutForest(AnomalyModel):
    """RobustRandomCutForest model combined with LDCD"""

    def __init__(self, number_of_trees=40, train_size=500,
                 queue_size=500):
        super().__init__()
        self.forest = [rrcf.RCTree() for _ in range(number_of_trees)]
        self.train_size = train_size
        self.queue_size = queue_size
        self.current_index = 0
        self.anomaly_scores_queue = []
        self.points_to_add_in_future = []

    def __add_point_in_forest(self, point, index):
        for tree in self.forest:
            tree.insert_point(point, index=index)

    def __calculate_anomaly_score(self, point, need_to_add_point_from_queue=True):
        score = 0
        if need_to_add_point_from_queue:
            point_tuple = self.points_to_add_in_future[0]
            self.points_to_add_in_future = self.points_to_add_in_future[1:]
            self.__add_point_in_forest(point_tuple[0], index=point_tuple[1])

        for tree in self.forest:
            if need_to_add_point_from_queue:
                tree.forget_point(self.current_index - self.queue_size - self.train_size + 1)
            tree.insert_point(point, index=self.current_index)
            new_codisp = tree.codisp(self.current_index)
            score += new_codisp / len(self.forest)
            tree.forget_point(self.current_index)
            assert len(tree.leaves) == self.train_size
        self.points_to_add_in_future.append((point, self.current_index))
        return score

    def predict_anomaly_proba(self, point)->float:
        self.current_index += 1
        if self.current_index <= self.train_size:
            self.__add_point_in_forest(point, self.current_index)
            return 0
        if self.current_index <= self.train_size + self.queue_size:
            need_to_add_point_from_queue = False
            score = self.__calculate_anomaly_score(point, need_to_add_point_from_queue)
            self.anomaly_scores_queue.append(score)
            return 0

        current_score = self.__calculate_anomaly_score(point)
        pvalue = 1 - np.sum(
            np.where(np.array(self.anomaly_scores_queue) > current_score, 1, 0)
        ) / self.queue_size

        self.anomaly_scores_queue.append(current_score)
        self.anomaly_scores_queue = self.anomaly_scores_queue[1:]
        assert len(self.anomaly_scores_queue) == self.queue_size
        return pvalue
