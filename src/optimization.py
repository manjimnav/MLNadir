import numpy as np

def get_distribution(model_name: str):
    distribution = {}
    if model_name == 'svr':
        distribution = dict(C=[0, 1, 2, 3, 4],
                     kernel=['linear', 'poly', 'rbf', 'sigmoid'])
    elif model_name == 'dummy':
        distribution = dict(strategy=['last', 'mean', 'drift'])
    elif model_name == 'lasso':
        distribution = dict(alpha=[0.0001, 0.001, 0.01, 0.1, 1, 2, 3])
    elif model_name == 'ridge':
        distribution = dict(alpha=[0.0001, 0.001, 0.01, 0.1, 1, 2, 3])
    elif model_name == 'elasticnet':
        distribution = dict(alpha=[0.0001, 0.001, 0.01, 0.1, 1, 2, 3],
                     l1_ratio=np.arange(0.1, 1, 0.1).tolist())
    elif model_name == 'decisiontree':
        distribution = dict(splitter=['best', 'random'],
                        max_depth=[i for i in range(1, 100, 2)],
                     criterion=['friedman_mse', 'squared_error', 'poisson', 'absolute_error'])
    elif model_name == 'randomforest':
        distribution = dict(n_estimators=[10, 50, 100, 200, 300, 400, 500],
                        max_depth=[i for i in range(1, 100, 2)],
                     criterion=['friedman_mse', 'squared_error', 'poisson', 'absolute_error'],
                     n_jobs = [-1])
    elif model_name == 'gbr':
        distribution = dict(n_estimators=[50, 100, 200, 300, 400, 500],
                        max_depth=[i for i in range(1, 10, 2)],
                        learning_rate = [0.003, 0.001, 0.0001],
                     loss=['squared_error', 'absolute_error', 'huber', 'quantile'],
                     criterion=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
    elif model_name == 'xgb':
        distribution = {
            'n_estimators':[10, 50, 100, 200, 300, 400, 500],
            'max_depth': [i for i in range(1, 100, 2)],
            'objective': ['reg:squarederror', 'reg:tweedie'],
            'booster': ['gbtree'],
            'eval_metric': ['rmse'],
            'eta': [i/10.0 for i in range(3,6)]
        }
    elif model_name == 'catboost':
        distribution = {'depth'   :[i for i in range(1, 10, 2)],
                  'learning_rate' : [0.003, 0.001, 0.0001],
                  'iterations'    : [10, 50, 100, 200, 300, 400, 500],
                  'l2_leaf_reg':  [0, 1, 2, 3, 4],
                 }
    elif model_name == 'lightgbm':
        distribution = { 'n_estimators': [10, 50, 100, 200, 300, 400, 500],'max_depth':[i for i in range(1, 100, 2)],
            'learning_rate' : [0.003, 0.001, 0.0001], 'reg_alpha':[0,0.01,0.03], 'linear_tree': [False]}
    elif model_name == 'mlp':
        distribution = {'num_layers': [i for i in range(1, 5)],  'num_units': [i for i in range(1, 128, 2)],
            'learning_rate_init' :  [0.003, 0.001, 0.0001], 'early_stopping': [True], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'solver': ['adam'],'activation':['relu']}
    
    return distribution