from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from AnomalyModelABS import AnomalyModel


def reorganize_as_matrix(x, size, shape):
    matrix_x = []
    for i in range(size):
        matrix_x.append(x[i:i + shape])
    return matrix_x


class KNNmodel(AnomalyModel):
    def __init__(self, dimension,
                 training_size=100, 
                 calibration_size=100,
                 vector_size=5,
                 neigbouhrs=10):
        self.query = []
        self.scores = []
        self.x_t = []
        self.x_c = []
        self.training_size = training_size
        self.calibration_size = calibration_size
        self.vector_size = vector_size
        self.neigbouhrs = neigbouhrs
        self.dimension = dimension
    
    def predict_anomaly_proba(self, point)->float:
        self.query += [*point.tolist()]
        if len(self.query)  < self.vector_size:
            return 0
        else:
            last_val = np.array(self.query[-self.vector_size:])
            self.query = self.query[1:]
            if len(self.x_t) < self.training_size:
                self.x_t.append(last_val)
                return 0
            else:
                nbrs = NearestNeighbors(n_neighbors=self.neigbouhrs, algorithm='auto').fit(self.x_t)
                distances, _ = nbrs.kneighbors([last_val])
                
                current_score = np.mean(distances)
                if len(self.x_t) == self.training_size and\
                   len(self.x_c) == self.calibration_size:
                    self.x_t.pop(0)
                    self.x_t.append(self.x_c.pop(0))

                if len(self.scores) == self.calibration_size:
                    pval = np.sum(
                        np.where(np.array(self.scores) < current_score, 1, 0)
                    ) / self.calibration_size
                    self.scores = self.scores[1:]
                else:
                    pval = 0

                self.x_c.append(last_val)
                self.scores.append(current_score)
                assert len(self.x_t) == self.training_size
                assert len(self.x_c) <= self.calibration_size
                return pval