import yaml
import importlib
import inspect
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostRegressor
import lightgbm as ltb

class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_layers=1, num_units=8, learning_rate_init=0.003, early_stopping=True, learning_rate='constant', solver='adam', activation='relu'):

        self.model = None

        self.num_layers = num_layers
        self.num_units = num_units
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.solver = solver
        self.activation = activation
        

    def fit(self, X, y):
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.num_units for _ in range(self.num_layers)),
            learning_rate_init = self.learning_rate_init,
            early_stopping=self.early_stopping,
            learning_rate=self.learning_rate,
            solver=self.solver,
            activation=self.activation,
            random_state=123
        )
        self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
def get_model(model_name:str):

    model = None
    if model_name == "mlp":

        model = MLPWrapper()  # Instantiates the model **model_params

    elif model_name not in ['xgb', 'catboost', 'lightgbm']:
        args = yaml.safe_load(open(f'config/baseline/{model_name}.yaml', 'r')) 
        
        model_name, import_module, model_params = args['name'], args['module'], {}
        """Returns a scikit-learn model."""
        model_class = getattr(importlib.import_module(import_module), model_name)
        
        model_inspect = inspect.signature(model_class.__init__)
        arguments = list(model_inspect.parameters.keys())
        
        if 'random_state' in arguments:
            model_params.update({'random_state':123})
        if 'n_jobs' in arguments:
            model_params.update({'n_jobs':-1})
        
        model = model_class(**model_params)  # Instantiates the model **model_params
    elif model_name=='xgb':
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=123)
    elif model_name=='catboost':
        model = CatBoostRegressor(random_state=123, loss_function= 'MultiRMSE', eval_metric= 'MultiRMSE',logging_level='Silent')
    elif model_name=='lightgbm':
        model = ltb.LGBMRegressor(random_state=123)
    

    return model