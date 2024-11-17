import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            # Сгенерировать рандомные индексы
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            # выбор сгенерированных индексов, также определение цели на основе сгенерированных индексов
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        # получение прогнозов и усреднение их
        predictions = np.zeros((len(data), len(self.models_list)))
        for i, model in enumerate(self.models_list):
            predictions[:, i] = model.predict(data)
        return np.mean(predictions, axis=1)

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        # Для каждой модели и каждой точки проверяем была ли точка OOB
        for model_idx, model in enumerate(self.models_list):
            # Получение индексов, которые не использованы в данном bag (OOB samples)
            oob_m = ~np.isin(np.arange(len(self.data)), self.indices_list[model_idx])
            oob_i = np.where(oob_m)[0]

            if len(oob_i) > 0:
                # получение прогноза для OOB samples
                oob_predict = model.predict(self.data[oob_i])

                # сохранение прогноза
                for idx, pred in zip(oob_i, oob_predict):
                    list_of_predictions_lists[idx].append(pred)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([
            np.mean(pred_list) if len(pred_list) > 0 else None
            for pred_list in self.list_of_predictions_lists
        ])


    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        # расчет MSE
        self._get_averaged_oob_predictions()
        mask = self.oob_predictions != None
        if not np.any(mask):
            return None
        return np.mean((self.oob_predictions[mask] - self.target[mask]) ** 2)