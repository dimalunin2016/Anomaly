import numpy as np
from tqdm import tqdm_notebook as tqdm
import scipy as sp
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import typing as tp
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.patches as mpatches
from AnomalyGenerator import Anomaly, AnomalyGenerator
from AnomalyModelRunner import run_model_on_generator, show_model_results
from itertools import chain
from AnomalyModelABS import AnomalyModel

def reorganize_as_matrix(x, size, shape):
    matrix_x = []
    for i in range(size):
        matrix_x.append(x[i:i + shape])
    return np.array(matrix_x)


class OutputNeuron():
    def __init__(self, shape, ni_size, mod, distribution):
        self.shape = shape
        self.ni_size = ni_size
        self.weight = np.zeros(self.ni_size)
        self.mod = mod
        self.M = 0
        self.time = 0
        self.distribution = distribution
        self.alpha = 0.1
        self.value = np.zeros(self.shape).reshape(-1, 1)
        
    def init_neuron(self, SNIID, time, means, stds, real_val, x):
        for j in range(self.ni_size):
            self.weight[SNIID[j]] = self.mod ** j
        self.M = 1
        self.time = time
        curr = x.copy()
        means = means.reshape(-1, 1)
        stds = stds.reshape(-1, 1)

        size = int(self.alpha * len(curr[0]))
        for i in range(self.shape):
            curr[i] = sorted(curr[i], key=lambda x: (x - real_val[i]) ** 2)
            means[i] = np.mean(curr[i][:size])
            stds[i] = np.std(curr[i][:size])
        self.value = self.distribution.rvs(self.shape).reshape(-1, 1) * stds + means
        
    def __str__(self):
        answer = f"""
OutputNeuron:
time: {self.time};
M: {self.M};
mod: {self.mod};
ni_size: {self.ni_size};
value: {str(self.value)};
weights: {self.weight};
======================

        """
        return answer

    
class eSNNAnomalyClassifierModified(AnomalyModel):
    def __init__(self, shape=1, ni_size=10, no_max_size=50,
                 window_size=50, ts=1000, sim=0.17, mod=0.6, C=0.4, 
                 value_correction_constant=0.9,
                 beta = 1.6,
                 distribution=sps.norm(loc=0, scale=1), min_pvalue=0.999,
                 errror_type='dynamically',
                 verbose=False):
        self.distribution = distribution
        self.shape = shape
        self.dimension = 1
        self.ni_size = ni_size
        self.no_max_size = no_max_size
        self.window_size = window_size
        self.ts = ts
        self.beta = beta
        self.sim = sim
        self.mod = mod
        self.C = C
        self.errror_type = errror_type
        if errror_type == 'dynamically' or errror_type == 'mean':
            self.errror_working_policy = np.mean
        if errror_type == 'min':
            self.errror_working_policy = np.min
        if errror_type == 'max':
            self.errror_working_policy = np.max
        self.value_correction_constant = value_correction_constant
        self.min_pvalue = min_pvalue
        self.verbose=verbose
        
        self.min_size = self.shape + self.window_size - 1 if self.dimension == 1 else \
                        self.dimension * self.window_size
        #self.min_size = self.dimension * self.window_size
        self.errors = None
        self.current_no_size = 0
        self.weights_sum = (1 - self.mod ** self.ni_size) / (1 - self.mod)
        self.gamma = self.C * (1 - self.mod ** (2 * self.ni_size)) / (1 - self.mod ** 2)
        self.x_values_list = []
        self.steam_values_window = []
        self.last_pvalue = 0
        self.start_predicting = False
        self.is_anomaly_array = []
        self.SNIID = np.zeros(self.ni_size, dtype=int)
        self.grf_mu = np.zeros(self.ni_size)
        self.grf_sigma = np.zeros(self.ni_size)
        self.max_dist = self.__calculate_max_distance()
        self.curr_index = 0
        self.output_neurons = []
        self.candidate_output_neuron = None
        self.stop = False
        self.curr_added = False
        self.curr_index = -1
        self.len_without_anomaly = 0


    def __calculate_max_distance(self):
        first_vector = []
        second_vector = []
        for i in range(self.ni_size):
            first_vector.append(self.mod ** self.ni_size - 1 - i)
            second_vector.append(self.mod ** i)
        max_dist = np.sqrt(np.sum((np.array(first_vector) - np.array(second_vector))** 2))
        return max_dist / self.ni_size
    
    
    def __before_prediction_start(self, X):
        current_values = np.array(X).T
        
        means = np.mean(current_values, axis=1)
        stds = np.std(current_values, axis=1)
        assert len(means) == self.shape
        assert len(stds) == self.shape
        predicted_values = self.distribution.rvs(
                            self.window_size * self.shape
                           ).reshape(
                                     self.shape, self.window_size
                             ) * stds.reshape(self.shape, 1) + means.reshape(self.shape, 1)
        assert predicted_values.shape == current_values.shape

        self.errors = np.abs((predicted_values - current_values) / (predicted_values + 1e-5)).T.tolist()
        self.is_anomaly_array = [0 for _ in range(self.window_size)]
        self.last_pvalue = 0
    

    def __init_GFRs(self, X):
        current_values = self.steam_values_window
        I_max = np.max(current_values)
        I_min = np.min(current_values)

        for j in range(self.ni_size):
            mu_curr = I_min + ((2 * j - 3) / 2) * ((I_max - I_min) / (self.ni_size - 2))
            sigma_curr = (1 / self.beta) * ((I_max - I_min) / (self.ni_size - 2))
            self.grf_mu[j] = mu_curr
            self.grf_sigma[j] = sigma_curr
        if self.verbose:
            print("self.grf_mu", self.grf_mu)
            print("self.grf_sigma", self.grf_sigma)

        
    def __calculate_spike_order(self, X):
        curr_x = self.steam_values_window[-1]
        firing_order_array = []
        for j in range(self.ni_size):
            exc = np.exp( -1 / 2 * ((curr_x - self.grf_mu[j]) / (self.grf_sigma[j] + 1e-5)) ** 2)
            T_j = self.ts * (1 - exc)
            firing_order_array.append((j, T_j))
        
        firing_order_array = sorted(firing_order_array, key=lambda x: x[1])
        assert np.unique(firing_order_array).shape[0] > 1
        self.SNIID = [int(x[0]) for x in firing_order_array]
        assert np.array(self.SNIID).shape[0] == self.ni_size


    def __init_new_output_neuron(self, X):
        self.candidate_output_neuron = OutputNeuron(shape=self.shape, 
                                                    ni_size=self.ni_size,
                                                    mod=self.mod,
                                                    distribution=self.distribution)
        current_values = np.array(X).T
        means = np.mean(current_values, axis=1)
        stds = np.std(current_values, axis=1)

        assert len(means) == self.shape
        assert len(stds) == self.shape
        
        self.candidate_output_neuron.init_neuron(self.SNIID, 
                                                 self.curr_index, means, stds,
                                                 X[-1].reshape(-1, 1), current_values)
        assert np.sum(np.abs(self.weights_sum - 
                             np.sum(self.candidate_output_neuron.weight))) < 1e-5
        
    
    def __find_similar(self):
        distances = []
        if self.current_no_size == 0:
            return None, None
        for i in range(self.current_no_size):
            distance = np.sqrt(
                    np.sum(
                        (self.output_neurons[i].weight - self.candidate_output_neuron.weight) ** 2)
                )
            curr_distance = distance / self.ni_size
            distances.append((i, curr_distance))
        ans = min(distances, key = lambda x: x[1])
        return ans

        
    def __first_output_neuron_fire(self):
        PSP = np.zeros(self.current_no_size)
        to_fire = []
        for order, input_neurons_index in enumerate(self.SNIID):
            for i in range(self.current_no_size):
                assert np.sum(np.abs(self.weights_sum - np.sum(self.output_neurons[i].weight))) < 1e-5
                PSP[i] += self.output_neurons[i].weight[input_neurons_index] * (self.mod ** order)
                fired = int(PSP[i] > self.gamma)
                if fired > 0:
                    to_fire.append((i, PSP[i]))
            if len(to_fire) > 0:
                return max(to_fire, key=lambda x: x[1])[0]
        return None
    
    
    def __update_neuron(self, index):
        n_s = self.output_neurons[index]
        n_s.weight = self.candidate_output_neuron.weight
        #(n_s.weight * n_s.M + self.candidate_output_neuron.weight) / (n_s.M + 1)
        n_s.value = (n_s.value * n_s.M + self.candidate_output_neuron.value) / (n_s.M + 1)
        n_s.time = (n_s.time * n_s.M + self.candidate_output_neuron.time) / (n_s.M + 1)
        n_s.M += 1
        assert np.sum(np.abs(self.weights_sum - np.sum(self.output_neurons[index].weight))) < 1e-5


    def __remove_oldest(self):
        min_t = self.curr_index + 1
        deleted_index = -1
        for i in range(self.current_no_size):
            if self.output_neurons[i].time < min_t:
                min_t = self.output_neurons[i].time
                deleted_index = i
        del self.output_neurons[deleted_index]
        self.output_neurons[deleted_index] = self.candidate_output_neuron
        return deleted_index
    
    def __value_correction(self, fired_index, y, x_last):
        self.output_neurons[fired_index].value += (x_last - y) * self.value_correction_constant
    
    
    def __classify_anomaly(self, last_error):
        curr_errors = []
        for error, anomaly in zip(self.errors[:-1], self.is_anomaly_array):
            if anomaly == 0:
                curr_errors.append(error)

        anomaly_size = int(2)
        if len(curr_errors) == 0 or np.sum(self.is_anomaly_array[-anomaly_size:]) == anomaly_size:
            self.is_anomaly_array.append(0)
            self.last_pvalue = 0
            self.len_without_anomaly = self.len_without_anomaly + 1
            return

        curr_errors = self.errror_working_policy(np.array(curr_errors), axis=1)

        if self.verbose:
            print("len curr errors:", curr_errors.shape)
        means = np.mean(curr_errors)
        stds = np.std(curr_errors)
        last_error = self.errror_working_policy(last_error)
        error_values = (last_error - means) / (stds + 1e-5)

        pval = self.distribution.cdf(error_values)
        if self.verbose:
            print(means, stds, error_values, checked_error, pval)
            
        is_anomaly = 1 if pval > self.min_pvalue else 0
        self.is_anomaly_array.append(is_anomaly)
        if is_anomaly == 1:
            self.len_without_anomaly = 0
        else:
            self.len_without_anomaly = self.len_without_anomaly + 1
        self.last_pvalue = pval
        
    
    def predict_anomaly_proba(self, point)->float:
        self.curr_index += 1
        self.curr_added = False
        self.x_values_list += [*point.tolist()]
        
        if len(self.x_values_list) < self.min_size + self.dimension:
            self.last_pvalue = 0
            return self.last_pvalue
        if len(self.x_values_list) >= self.min_size + self.dimension:

            self.x_values_list = self.x_values_list[self.dimension:]
            X = reorganize_as_matrix(self.x_values_list, self.window_size, self.shape)
            self.steam_values_window = np.array(X).T[-1]
            if self.verbose:
                print("X:", X[-1])
            
            if not self.start_predicting:
                self.__before_prediction_start(X)
                self.start_predicting = True
                self.last_pvalue = 0
                return self.last_pvalue
            
            
            self.errors = self.errors[1:]
            self.is_anomaly_array = self.is_anomaly_array[1:]
            assert len(self.errors) == self.window_size - 1, f"{len(self.errors)} != {self.window_size - 1}"
            assert len(self.is_anomaly_array) == self.window_size - 1, \
            f"{len(self.is_anomaly_array)} != {self.window_size - 1}"
            
            self.__init_GFRs(X)
            self.__calculate_spike_order(X)
            self.__init_new_output_neuron(X)
            
            output_similar_neuron_index, distance = self.__find_similar()
            
            if self.verbose:
                print("new out", self.candidate_output_neuron)
                print("DISTANCE COMPARE", distance, self.sim * self.max_dist, self.max_dist)

            if distance is not None and distance < self.sim * self.max_dist:
                if self.verbose:
                    print("OLD near", self.output_neurons[output_similar_neuron_index])
                self.__update_neuron(output_similar_neuron_index)
                if self.verbose:
                    print("updated_old_by_new")

            elif self.current_no_size < self.no_max_size:
                self.output_neurons.append(self.candidate_output_neuron)
                self.current_no_size += 1
                self.curr_added = True
                self.curr_index = self.current_no_size - 1
                if self.verbose:
                    print("ADDED")
            else:
                self.curr_index = self.__remove_oldest()
                self.curr_added = True

            ##################################
            
            if self.curr_added:
                fired_no_index = self.curr_index
            else:
                fired_no_index = self.__first_output_neuron_fire()
                

            if fired_no_index is None:
                if self.verbose:
                    print("fired_no_index is None")
                self.errors.append([-1 for _ in range(self.shape)])
                self.is_anomaly_array.append(1)
                self.last_pvalue = 1
                self.len_without_anomaly = 0
                return self.last_pvalue


            if self.verbose:
                print("fired_no_index", fired_no_index)
                print("Fired neuron", self.output_neurons[fired_no_index])


            curr_y = self.output_neurons[fired_no_index].value.copy()
            self.errors.append((np.abs(curr_y.squeeze() - X[-1]) / (curr_y.squeeze() + 1e-5)).T.tolist())
            
            if self.verbose:
                print("NO number:", self.current_no_size)
                print("X values", self.x_values_list)
                print("Errors", self.errors)
                print("anomaly array", self.is_anomaly_array)

            assert len(self.errors) == self.window_size
            assert len(self.is_anomaly_array) == self.window_size - 1
            
            last_error = self.errors[-1]
            if self.verbose:
                print("Current Error", last_error)

            if self.errror_type == 'dynamically':
                if self.len_without_anomaly < 3 * self.window_size:
                    self.errror_working_policy = np.min
                elif self.len_without_anomaly < 8 * self.window_size:
                    self.errror_working_policy = np.mean
                else:
                    self.errror_working_policy = np.max
                  
            self.__classify_anomaly(last_error)

            if self.is_anomaly_array[-1] == 0:
                self.__value_correction(fired_no_index, curr_y, X[-1].reshape(-1, 1))
            
            assert len(self.x_values_list) == self.min_size
            return self.last_pvalue