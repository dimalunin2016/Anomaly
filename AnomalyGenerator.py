import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as tp
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.patches as mpatches


class Anomaly:
    """Types of anomalies"""

    novelty = "novelty"
    collective = "collective"
    individual = "individual"


class AnomalyGenerator:
    """Add anomalies to time series data"""

    def __init__(self):
        super().__init__()
        self.current_index_to_yield = 0
        self.anomalies = {}
        self.anomalies_has_already_applied = False
        self.dataframe = None

    def __prepare_table(self, time_column, type_of_time_column) -> None:
        """Prepare dataframe for comfort work with it
        :param time_column: name of time column
        :param type_of_time_column: "float" or "timeseries", it is needed to show right xlabels on plots
        :return:
        """
        
        if type_of_time_column == "timeseries":
            try:
                self.dataframe[time_column] = pd.to_datetime(self.dataframe[time_column])
                self.is_time_series_column = True
            except Exception:
                raise ValueError("Time column is not time series")
        elif type_of_time_column == "float":
            try:
                self.dataframe[time_column] = self.dataframe[time_column].astype(float)
                self.is_time_series_column = False
            except Exception:
                raise ValueError("Time column is not float")
        else:
            raise ValueError("Uknown type Time column")
        self.dataframe = self.dataframe.sort_values(time_column)
        self.data_columns = set(self.dataframe.columns) - {time_column}
        self.time_column = time_column
        self.size = self.dataframe.shape[0]
        self.dataframe.index = np.arange(0, self.size)
        self.anomalies_has_already_applied = False
        for column in self.data_columns:
            try:
                self.dataframe[column] = self.dataframe[column].astype(float)
            except Exception:
                raise ValueError(f"Column {column}: all columns must be int or float")
        self.dataframe.fillna(0, inplace=True)
        self.start_dataframe = self.dataframe.copy()

    def read_data_from_file(self, name_of_file, 
                            time_column, sep=',',
                            type_of_time_column="timeseries") -> 'AnomalyGenerator':
        """Constructs data from file without anomalies
        :param name_of_file: file with data
        :param time_column: name of time column
        :param sep: separator for read_csv
        :param type_of_time_column: "float" or "timeseries", it is needed to show right xlabels on plots
        :return: constructed from AnomalyGenerator
        """
        self.dataframe = pd.read_csv(name_of_file, sep=sep)
        self.__prepare_table(time_column, type_of_time_column)
        return self

    def read_data_from_pd_dataframe(self, dataframe, 
                                    time_column,
                                    type_of_time_column="timeseries") -> 'AnomalyGenerator':
        """Constructs data from dataframe without anomalies
        :param dataframe: constructing from dataframe
        :param time_column: name of time column
        :param type_of_time_column: "float" or "timeseries", it is needed to show right xlabels on plots
        :return: constructed from AnomalyGenerator
        """
        self.dataframe = dataframe.copy()
        self.__prepare_table(time_column, type_of_time_column)
        return self
    
    @staticmethod
    def __iterate_by_two_elements(array):
        """Generator for iterating by 2 elements from array
        :param array: array for iterating
        :return: two elements of array
        """
        for i in range(len(array) - 1):
            yield array[i], array[i + 1]

    @staticmethod
    def __get_params_for_new_anomaly(current_anomaly_place,
                                     next_anomaly_place,
                                     types_of_outliers_for_columns,
                                     anomaly_minsize,
                                     anomaly_maxsize,
                                     generator,
                                     previous_novelty_in_columns,
                                     data_size) -> tp.Dict[str, tp.Any]:
        """Constructs parameters for new anomaly (which already has place)
        :param current_anomaly_place: start place of anomaly
        :param next_anomaly_place: start place of next anomaly
        :param types_of_outliers_for_columns: dict of which types of anomalies are able for each column
        :param anomaly_minsize: minimum size of anomaly length
        :param anomaly_maxsize: maximum size of anomaly length
        :param generator: generator to generate random
        :param previous_novelty_in_columns: previous novelty start place in this column
        :param data_size: size of all data
        :return: return parameters for new anomaly
        """
        anomaly_params = dict()
        anomaly_params["start_index"] = current_anomaly_place
        possible_columns = list(types_of_outliers_for_columns.keys())
        max_size_anomaly_window = min(
            (next_anomaly_place - current_anomaly_place), anomaly_maxsize)
        anomaly_window_size = generator.randint(low=anomaly_minsize, high=max_size_anomaly_window)
        
        anomaly_params["size"] = anomaly_window_size
        anomaly_params["end_index"] = anomaly_window_size + current_anomaly_place
        
        number_of_columns_with_anomalies = generator.randint(low=1, high=len(possible_columns) + 1)
        anomaly_columns = generator.choice(possible_columns, size=number_of_columns_with_anomalies, replace=False)
        
        anomalies_in_columns = {}
        for column in anomaly_columns:
            anomalies_in_columns[column] = generator.choice(
                types_of_outliers_for_columns[column])
            if anomalies_in_columns[column] == Anomaly.novelty:
                if current_anomaly_place - previous_novelty_in_columns[column] < 0.33 * data_size:
                    anomalies_in_columns[column] = generator.choice(
                        list(set(types_of_outliers_for_columns[column]) - {Anomaly.novelty}))
                else:
                    previous_novelty_in_columns[column] = current_anomaly_place
                
        anomaly_params["anomalies_in_columns"] = anomalies_in_columns
        return anomaly_params
        
    def __add_types_of_anomalies(self, anomaly_places, 
                                 types_of_outliers_for_columns) -> None:
        """Constructs parameters for all anomalies with fixed places
        :param anomaly_places: array of start places of anomalies
        :param types_of_outliers_for_columns: dict of which types of anomalies are able for each column
        """
        self.anomalies = []
        previous_novelty_in_columns = {column: -self.size for column in self.data_columns}
        for current_anomaly_place, next_anomaly_place in \
                AnomalyGenerator.__iterate_by_two_elements(anomaly_places + [self.size]):
            self.anomalies.append(
                AnomalyGenerator.__get_params_for_new_anomaly(
                    current_anomaly_place, next_anomaly_place,
                    types_of_outliers_for_columns,
                    self.anomaly_minsize,
                    self.anomaly_maxsize,
                    self.generator,
                    previous_novelty_in_columns,
                    self.size))
    
    def __find_anomaly_places(self, distance_between_anomalies,
                              length_of_normal_data_in_start) -> tp.List[int]:
        """Find sutable places for anomalies
        :param length_of_normal_data_in_start: length of normal data in the beginning of dataset
        :param distance_between_anomalies: minimum distance between start of anomalies
        :return: sorted list of anomaly starts
        """
        from_ind = length_of_normal_data_in_start
        to_ind = self.size - self.anomaly_maxsize
        places = set()
        intervals = [(from_ind, to_ind)]
        for ind in range(self.number_of_anomalies):
            idx = self.generator.choice(len(intervals))
            interval = intervals[idx]
            intervals.remove(interval)
            value = self.generator.randint(low=interval[0], high=interval[1])
            places.add(value)
            interval_new_a = (interval[0], value - distance_between_anomalies)
            size_a = value - interval[0] - distance_between_anomalies
            interval_new_b = (value + distance_between_anomalies, interval[1])
            size_b = interval[1] - value - distance_between_anomalies
            if size_a > 0:
                intervals.append(interval_new_a)
            if size_b > 0:
                intervals.append(interval_new_b)
        return sorted(list(places))

    def __init_all_params_of_anomalies(self, types_of_outliers_for_columns, *,
                                       length_of_normal_data_in_start=1000,
                                       pct_of_outliers=0.005,
                                       distance_between_anomalies=None,
                                       number_of_anomalies=None,
                                       minsize=10,
                                       random_state=42) -> None:
        """Constructs all anomaly parameters
        :param types_of_outliers_for_columns: dict of which types of anomalies are able for each column
        :param length_of_normal_data_in_start: length of normal data in the beginning of dataset
        :param pct_of_outliers: percentile of anomaly data
        :param distance_between_anomalies: minimum distance between start places of outliers
        :param number_of_anomalies: fixed number of anomalies in dataset (in this case pct_of_outliers is unuseful)
        :param minsize: minimum size of anomaly length
        :param random_state: random seed
        """
        self.anomaly_minsize = minsize
        self.anomaly_maxsize = 2 * minsize
        self.number_of_anomalies = int((self.size / self.anomaly_maxsize) * pct_of_outliers)
        self.generator = np.random.RandomState(random_state)
        if number_of_anomalies is not None:
            self.number_of_anomalies = number_of_anomalies

        if distance_between_anomalies is None:
            distance_between_anomalies = int((self.size - length_of_normal_data_in_start)
                                             / (self.number_of_anomalies * 2))

        anomaly_places = self.__find_anomaly_places(distance_between_anomalies, length_of_normal_data_in_start)
        self.__add_types_of_anomalies(anomaly_places, types_of_outliers_for_columns)
        assert self.number_of_anomalies == len(self.anomalies)
  
    def __prepare_collective_anomalies(self, smoothing_level=0.01) -> None:
        """Constructs model to generate collective anomalies
        :param smoothing_level: parameter for SimpleExpSmoothing
        """
        self.collective_anomalies_models = {}
        for column in self.data_columns:
            self.collective_anomalies_models[column] = SimpleExpSmoothing(self.dataframe[column]) \
                .fit(smoothing_level=smoothing_level, optimized=False)
    
    def __apply_new_collective_anomaly(self, column, 
                                       start_anomaly_place,
                                       end_anomaly_place) -> None:
        """Add to data one new collective anomaly
        :param column: column with anomaly
        :param start_anomaly_place: starting place of anomaly
        :param end_anomaly_place: ending place of anomaly
        """
        anomaly_model = self.collective_anomalies_models[column]
        new_values = anomaly_model.predict(start_anomaly_place, 
                                           end_anomaly_place)
        self.dataframe[column].loc[start_anomaly_place:
                                   end_anomaly_place] = new_values

    def __apply_new_novelty_anomaly(self, column, 
                                    start_anomaly_place,
                                    end_anomaly_place) -> None:
        """Add to data one new novelty
        :param column: column with anomaly
        :param start_anomaly_place: starting place of novelty
        :param end_anomaly_place: ending place of novelty
        """
        size = end_anomaly_place - start_anomaly_place
        mean_before = (np.max(self.start_dataframe[column].iloc[:start_anomaly_place]) - 
                       np.min(self.start_dataframe[column].iloc[:start_anomaly_place]))
        adding = self.generator.choice([-1, 1]) * self.generator.uniform(mean_before * 0.4, mean_before * 0.55)
        for ind, place in enumerate(range(start_anomaly_place, end_anomaly_place)):
            self.dataframe[column].loc[place] += adding * (ind / size)
        self.dataframe[column].loc[end_anomaly_place:] += adding
    
    def __apply_all_novelty_anomalies(self) -> None:
        """Constructs all novelties"""

        for anomaly in self.anomalies:
            column_anomalies = anomaly["anomalies_in_columns"]
            for column in column_anomalies:
                if column_anomalies[column] == Anomaly.novelty:
                    self.__apply_new_novelty_anomaly(column, 
                                                     anomaly['start_index'],
                                                     anomaly['end_index'])

    def __apply_new_individual_anomaly(self, column, 
                                       start_anomaly_place,
                                       end_anomaly_place) -> None:
        """Add to data one new individual anomaly
        :param column: column with anomaly
        :param start_anomaly_place: starting place of anomaly
        :param end_anomaly_place: ending place of anomaly
        """
        size = end_anomaly_place - start_anomaly_place
        middle = (start_anomaly_place + end_anomaly_place) // 2
        adding = self.generator.choice([-1, 1]) * \
            ((self.generator.rand() + 1) * np.max(abs(self.start_dataframe[column].iloc[:start_anomaly_place])) -
             abs(self.start_dataframe[column].iloc[middle]))
        
        for ind, place in enumerate(range(start_anomaly_place, end_anomaly_place)):
            if place < middle:
                self.dataframe[column].loc[place] += adding * (ind / (size / 2))
            else:
                self.dataframe[column].loc[place] += adding * ((size - ind) / (size / 2))
    
    def __apply_consructed_anomalies_to_data(self, smoothing_level_for_context_anomaly=0.01) -> None:
        """Add all anomalies with parameters to data
        :param smoothing_level_for_context_anomaly: parameter to create collective anomaly
        """
        if self.anomalies_has_already_applied:
            return
        self.__apply_all_novelty_anomalies()
        self.__prepare_collective_anomalies(smoothing_level_for_context_anomaly)
        self.anomalies_has_already_applied = True
        for anomaly in self.anomalies:
            start_place = anomaly['start_index']
            end_place = anomaly['end_index']
            column_anomalies = anomaly["anomalies_in_columns"]
            for column in column_anomalies:
                if column_anomalies[column] == Anomaly.collective:
                    self.__apply_new_collective_anomaly(column, start_place,
                                                        end_place)
                if column_anomalies[column] == Anomaly.individual:
                    self.__apply_new_individual_anomaly(column, start_place,
                                                        end_place)

    def get_size(self) -> int:
        return self.size
    
    def add_anomalies(self, types_of_outliers_for_columns, *,
                      length_of_normal_data_in_start=1000,
                      pct_of_outliers=0.005,
                      distance_between_anomalies=None,
                      number_of_anomalies=None,
                      minsize=10,
                      random_state=42,
                      smoothing_level_for_context_anomaly=0.01) -> None:
        """Constructs and applies anomaly to data, but do it once
        :param types_of_outliers_for_columns: dict of which types of anomalies are able for each column
        (types must be parameters from Anomaly class)
        :param length_of_normal_data_in_start: length of normal data at the beginning of dataset
        :param pct_of_outliers: percentile of anomaly data
        :param distance_between_anomalies: minimum distance between start places of outliers
        :param number_of_anomalies: fixed number of anomalies in dataset (in this case pct_of_outliers is unuseful)
        :param minsize: minimum size of anomaly length
        :param random_state: random seed
        :param smoothing_level_for_context_anomaly: parameter for SimpleExpSmoothing for collective anomalies
        """
        if self.anomalies_has_already_applied:
            return 
        self.__init_all_params_of_anomalies(types_of_outliers_for_columns,
                                            length_of_normal_data_in_start=length_of_normal_data_in_start,
                                            pct_of_outliers=pct_of_outliers,
                                            distance_between_anomalies=distance_between_anomalies,
                                            number_of_anomalies=number_of_anomalies,
                                            minsize=minsize,
                                            random_state=random_state)
        self.__apply_consructed_anomalies_to_data(
            smoothing_level_for_context_anomaly=smoothing_level_for_context_anomaly)

    def reset(self) -> None:
        """Removes all anomalies from data"""

        self.current_index_to_yield = 0
        self.anomalies = {}
        self.anomalies_has_already_applied = False
        self.dataframe = self.start_dataframe.copy()
    
    def show_anomaly(self, column_name_to_draw, index_of_anomaly,
                     predicted_anomaly_points=None) -> None:
        """Draw data near anomaly
        :param column_name_to_draw: column to draw
        :param index_of_anomaly: index of anomaly (from 0)
        :return: plot with data
        """
        try:
            anomaly = self.anomalies[index_of_anomaly]
        except Exception:
            raise ValueError("Sorry, I don't have anomaly with this index")
        start = anomaly["start_index"]
        end = anomaly["end_index"]
        size = anomaly["size"]
        size = max(size, 100)
        self.show_data(column_name_to_draw,
                       draw_between=(max(start - size * 5, 0), min(end + size * 5, self.size)),
                       predicted_anomaly_points=predicted_anomaly_points)
        
    def get_new_data(self) -> tp.Any:
        """Generate new data with anomaly (anomalies must be added before)"""

        for ind in range(self.size):
            self.current_index_to_yield = ind
            yield self.dataframe[self.data_columns].iloc[ind]

    def get_time_values(self, draw_between=None) -> np.ndarray:
        """Get values of time_column"""
        start_index = 0
        last_index = self.size
        if draw_between is not None:
            start_index, last_index = draw_between
        df = self.dataframe.iloc[start_index:last_index]
        time_values = np.array(df.index)
        if self.is_time_series_column:
            time_values = np.array(df[self.time_column])
        return time_values

    def __plot_anomaly_intervals(self, column_name_to_draw,
                                 current_df,
                                 draw_between,
                                 handles) -> None:
        colors = {Anomaly.novelty: "cyan",
                  Anomaly.individual: "red",
                  Anomaly.collective: "black",
                  None: "lime"}
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Individual anomaly')
        cyan_patch = mpatches.Patch(color='cyan', alpha=0.3, label='Novelty')
        black_patch = mpatches.Patch(color='black', alpha=0.3, label='Collective anomaly')
        lime_patch = mpatches.Patch(color='lime', alpha=0.3, label='Anomaly not in this column')
        patches = {"cyan": cyan_patch,
                   "black": black_patch,
                   "lime": lime_patch,
                   "red": red_patch}
        curr_handles = set()
        time_values = self.get_time_values(draw_between)
        for anomaly in self.anomalies:
            start = anomaly["start_index"]
            end = anomaly["end_index"]
            if start >= current_df.index[0] and end <= current_df.index[-1]:
                color = colors[anomaly["anomalies_in_columns"].get(column_name_to_draw)]
                plt.axvspan(time_values[start - current_df.index[0]],
                            time_values[end - current_df.index[0]], alpha=0.3, color=color)
                curr_handles.add(patches[color])
        if curr_handles:
            handles.update(curr_handles)

    @staticmethod
    def __plot_predicted_anomaly_dots(start_index, last_index,
                                      time_values, data,
                                      predicted_anomaly_points,
                                      handles) -> None:

        predicted_anomaly_points = np.array(predicted_anomaly_points)
        predicted_anomaly_points = predicted_anomaly_points[
            np.where((predicted_anomaly_points > start_index) &
                     (predicted_anomaly_points < last_index))[0]
        ]
        predicted_anomaly_points -= start_index
        dots = plt.scatter(time_values[predicted_anomaly_points],
                           data[predicted_anomaly_points], color='red', lw=3, zorder=2,
                           label="Model anomaly prediction")
        handles.add(dots)

    def show_data(self, column_name_to_draw, draw_between=None,
                  predicted_anomaly_points=None) -> None:
        """Plot data dependecy from time
        :param column_name_to_draw: column to draw
        :param draw_between: interval, which ids needed to plot (start index, end index)
        :param predicted_anomaly_points: indexes of points, which anomalies were predicted by model
        :return: plot with data
        """
        start_index = 0
        last_index = self.size
        if draw_between is not None:
            start_index, last_index = draw_between
        df = self.dataframe.iloc[start_index:last_index]
        data = np.array(df[column_name_to_draw])

        plt.figure(figsize=(20, 10))
        plt.title(f"The dependence of '{column_name_to_draw}' from time",
                  fontsize=15)
        time_values = self.get_time_values(draw_between)
        plt.plot(time_values, data, lw=1, color="blue", zorder=1)
        handles = set()
        if predicted_anomaly_points is not None:
            AnomalyGenerator.__plot_predicted_anomaly_dots(start_index, last_index,
                                                           time_values, data,
                                                           predicted_anomaly_points,
                                                           handles)
        if self.anomalies_has_already_applied:
            self.__plot_anomaly_intervals(column_name_to_draw, df, draw_between, handles)
        plt.legend(handles=sorted(handles, key=lambda x: x.get_label()), fontsize=15)
        plt.ylabel(f"Values of '{column_name_to_draw}'", fontsize=15)
        plt.xlabel("Time", fontsize=15)
