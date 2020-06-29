import typing as tp
import numpy as np


class NAB:

    def __init__(self, a_tp: float = 1, a_fp: float = -1, a_fn: float = -1, function_const: float = 5) -> None:
        """Init scorer
        :param a_tp: weight of true positive prediction
        :param a_fp: negative weight of false positive prediction
        :param a_fn: negative weight of missing anomaly
        :param function_const: function halfness (closer to 0 therefore a flatter function)
        """
        super().__init__()
        self.A_tp = a_tp
        self.A_fp = a_fp
        self.A_fn = a_fn
        self.function_const = function_const

    def __calculate_anomaly_score_for_one_anomaly(self, anomaly: tp.Dict[str, tp.Any], predict_anomaly_index: int) -> float:
        """Calculate score for one anomaly and one prediction
        :param anomaly: dict with all anomaly information
        :param predict_anomaly_index: index in data which is predicted as anomaly
        :return: score of current prediction
        """
        if anomaly["start_index"] > predict_anomaly_index:
            return self.A_fp
        y = predict_anomaly_index - anomaly["end_index"]
        if y > 0:
            const = -self.A_fp
        else:
            const = self.A_tp
        return const * (2 / (1 + np.exp(self.function_const * y)) - 1)

    def __calculate_raw_score(self, anomalies_raw: tp.List[tp.Any],
                              predict_anomaly_indexes_raw: tp.List[tp.Any]) -> float:
        """Calculate summary prediction score for one file
        :param anomalies_raw: list of anomalies in file (where one anomaly is dict)
        :param predict_anomaly_indexes_raw: list of indexes, where anomaly was predicted
        :return: score for the whole file
        """
        predict_anomaly_indexes = sorted(predict_anomaly_indexes_raw)
        anomalies = sorted(anomalies_raw, key=lambda x: x["start_index"])
        if len(predict_anomaly_indexes) > 0:
            max_not_anomaly_index = predict_anomaly_indexes[-1] * 100
        else:
            max_not_anomaly_index = 0
        anomalies = [{'start_index': -1000, 'size': 0, 'end_index': -1000,
                      'anomalies_in_columns': {}}] + anomalies + \
                    [{'start_index': max_not_anomaly_index, 
                      'size': 0, 'end_index': max_not_anomaly_index,
                      'anomalies_in_columns': {}}]
        it_anomalies = 0
        it_predicts = 0

        score = 0
        visited_anomalies = {0, len(anomalies)}
        while True:
            if it_predicts >= len(predict_anomaly_indexes):
                break
            predict_anomaly_index = predict_anomaly_indexes[it_predicts]
            anomaly = anomalies[it_anomalies]

            if predict_anomaly_index < \
                anomalies[it_anomalies + 1]["start_index"]:
                curr_score = \
                self.__calculate_anomaly_score_for_one_anomaly(anomaly, 
                                                               predict_anomaly_index)
                y = predict_anomaly_index - anomaly["end_index"]
                if y > 0 or (curr_score >= 0 and 
                                      it_anomalies not in visited_anomalies):
                    score += curr_score
                if y <= 0 and it_anomalies not in visited_anomalies:
                    visited_anomalies.add(it_anomalies)
                it_predicts += 1
                continue
            else:
                it_anomalies += 1
                continue
        score += (len(anomalies) - len(visited_anomalies)) * self.A_fn
        return score
    
    def score(self, list_of_anomalies_in_files: tp.List[tp.Any],
              list_of_predictions_for_files: tp.List[tp.Any]) -> float:
        """ Calculate score for all given files
        :param list_of_anomalies_in_files: list of lists with anomaly information in different datasets
        :param list_of_predictions_for_files: list of lists with prediction anomaly indexes in different datasets
        :return: normalizied summary score for predictions in different datasets
        """
        score_prediction = 0
        score_null = 0
        score_perfect = 0
        for anomaly, prediction in zip(list_of_anomalies_in_files,
                                       list_of_predictions_for_files):
            score_prediction += self.__calculate_raw_score(anomaly, prediction)
            perfect_prediction = [x['start_index'] for x in anomaly]
            score_null += self.__calculate_raw_score(anomaly, [])
            score_perfect += self.__calculate_raw_score(anomaly, perfect_prediction)
        score = 100 * (score_prediction - score_null) / (score_perfect - score_null)
        return score
