from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from AnomalyModelABS import AnomalyModel


class KNNmodel(AnomalyModel):
    def __init__(self, dimension,
                 training_size=100, 
                 calibration_size=100,
                 vector_size=5,
                 neigbouhrs=10):
        self.query = []
        self.training_size = training_size
        self.calibration_size = calibration_size
        self.vector_size = vector_size
        self.neigbouhrs = neigbouhrs
        self.dimension = dimension
    
    def predict_anomaly_proba(self, point)->float:
        self.query += [*point.tolist()]
        if len(self.query) == self.vector_size * \
                         self.dimension * (self.calibration_size + self.training_size):
            X = np.array(self.query).reshape(-1, self.vector_size * self.dimension)
            self.query = self.query[self.dimension:]
            train = X[:self.training_size]
            test = X[self.training_size:]
            nbrs = NearestNeighbors(n_neighbors=self.neigbouhrs, algorithm='auto').fit(train)
            distances, _ = nbrs.kneighbors(test)
            metrics = np.sum(distances, axis=1) / self.neigbouhrs
            current_score = metrics[-1]
            assert(metrics.shape[0] == self.calibration_size)
            pval = 1 - np.sum(
                np.where(np.array(metrics) > current_score, 1, 0)
            ) / self.calibration_size
            return pval
        else:
            return 0