import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time
from tqdm.notebook import tqdm
from skopt import BayesSearchCV
from IPython.display import display, clear_output
from .optimization import get_distribution
from .modeling import get_model
from .metrics import generate_results

def train(data, model_id, without_interval=True):
    
    not_included_cols = ['target', 'code', 'pred_time']
    
    if without_interval:
        not_included_cols.append('interval')
        
    input_columns = [col for col in data.columns if col not in not_included_cols]
    X = data[input_columns].values
    y = data['target'].values
    metadata = data['code'].values

    total_results = pd.DataFrame()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    
    model = get_model(model_id)
    distribution = get_distribution(model_id)
    
    training_time = time.time()
    opt = BayesSearchCV(model,
                        distribution,
                        n_iter=20,
                        cv=skf.split(X, metadata),
                        random_state=123,
                        scoring='neg_mean_absolute_error',
                        n_jobs=1
                        )
    training_time = (time.time()-training_time)/(60)
    
    opt.fit(X, y)
    
    pd.DataFrame(opt.cv_results_).to_csv('opt.csv')
    best_params = opt.best_params_

    for fold, (train_index, test_index) in tqdm(enumerate(skf.split(X, metadata))):
        
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        
        model = get_model(model_id)
        
        model.set_params(**best_params)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        local_results = generate_results(y_test, y_pred, data.iloc[test_index], fold, model_id, training_time, opt.best_params_)

        total_results = pd.concat((total_results, local_results))
    
    return total_results

def start_experimentation(data, model_identifiers, without_interval=True):
    
    results = pd.DataFrame()
    for model_id in tqdm(model_identifiers):
        print('traininig...')
        experiment_results = train(data, model_id, without_interval)
        results = pd.concat((results, experiment_results))
        clear_output()
        results['diff'] = (results.true-results.pred).abs()
        display(pd.DataFrame(results.groupby(['model'])['diff'].mean()))
    return results
        