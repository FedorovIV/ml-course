import numpy as np

import numpy as np
from scipy.stats import laplace

class LaplaceDistribution:
    
    def __init__(self, features: np.ndarray):
        '''
        Инициализация параметров распределения Лапласа — медианы и шкалы.
        
        Args:
            features: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        self.loc = np.median(features, axis=0)
        self.scale = self.mean_abs_deviation_from_median(features)

    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Вычисляет среднее абсолютное отклонение от медианы для каждого признака.
        
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        
        Returns:
        - mad: A numpy array of shape (n_features,) with the MAD for each feature.
        '''
        medians = np.median(x, axis=0)
        mad = np.mean(np.abs(x - medians), axis=0)
        return mad

    def logpdf(self, values):
        '''
        Возвращает логарифм плотности вероятности для каждого входного значения.
        
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        
        Returns:
        - log_prob: A numpy array of shape (n_objects, n_features) with the log probability densities.
        '''
        log_prob = -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        return log_prob
    
    def pdf(self, values):
        '''
        Возвращает плотность вероятности для каждого входного значения.
        
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        
        Returns:
        - prob: A numpy array of shape (n_objects, n_features) with the probability densities.
        '''
        return np.exp(self.logpdf(values))