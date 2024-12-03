import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        self.indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        self.data = data
        self.target = target
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain len(data) number of elements!'
        self.models_list = []
        for indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag))

        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        predictions = np.zeros((data.shape[0], self.num_bags))
        for i, model in enumerate(self.models_list):
            predictions[:, i] = model.predict(data)
        return np.mean(predictions, axis=1)
    
    def _get_oob_predictions_from_every_model(self):
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for model_idx, indices in enumerate(self.indices_list):
            oob_indices = set(range(len(self.data))) - set(indices)
            for i in oob_indices:
                prediction = self.models_list[model_idx].predict(self.data[i].reshape(1, -1))
                list_of_predictions_lists[i].append(prediction[0])
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([
            np.mean(pred_list) if len(pred_list) > 0 else None 
            for pred_list in self.list_of_predictions_lists
        ])
        
    def OOB_score(self):
        self._get_averaged_oob_predictions()
        mask = np.array([x is not None for x in self.oob_predictions])
        valid_predictions = self.oob_predictions[mask]
        valid_targets = self.target[mask]
        return mean_squared_error(valid_targets, valid_predictions)