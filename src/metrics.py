import pandas as pd
import glob
import re

def generate_results(y_test, y_pred, data_test, fold, model_id, duration, params=None):
    
    data = {'true': y_test, 'pred': y_pred, 'code': data_test['code'], 'interval': data_test['interval']}
    
    results = pd.DataFrame(data)
    
    results['model'] = model_id
    results['duration'] = duration
    results['fold'] = fold
    if params:
        results['params'] = str(params)
    else:
        results['params'] = str({})
    
    return results

def summarize_results(filenames='results/100ms/Scenario*.csv', groupby_cols = ['model']):
    total_results = pd.DataFrame()
    grouped_metrics = pd.DataFrame()
    for file in glob.glob(filenames):
        file = file.replace("\\", "/")
        results = pd.read_csv(file)
        scenario = re.findall(r'results/100ms/(Scenario\d+)_.*', file)[0]
        results['Time (s)'] = (results['interval']*0.1+1.1).round(1)
        results['case'] = results.code.apply(lambda x: x[:3] if len(x)<=6 else x[:5])
        results['scenario'] = scenario
                
        scenario_num = int(re.findall(r'results/100ms/Scenario(\d+)_.*', file)[0])

        total_results = pd.concat((total_results, results))
            
        for grouped_variables, group in results.groupby(groupby_cols[0] if len(groupby_cols)==1 else groupby_cols):
            
            diff = (group['true']-group['pred'])
            
            #diff = diff[diff<1]

            mae = diff.abs().mean()
            mae_std = diff.abs().std()

            mse = (diff**2).mean()
            mse_std = (diff**2).std()

            mape = (diff/group['true']).abs().mean()*100
            mape_std = (diff/group['true']).abs().std()*100
            
            #metrics = pd.DataFrame({'scenario': scenario, 'number':scenario_num, 'model': model, 'mae': f'{mae:.2E}', 'mse': f'{mse:.2E}', 'mape': f'{mape:.2E}', 'samples': len(group), 'params': group.iloc[0].params}, index=[0])
            grouped_variables = list(grouped_variables) if type(grouped_variables) == tuple else [grouped_variables]
            data_dictionary = {'scenario': scenario, 'number':scenario_num, 'mae': f'{mae:.2E} ± {mae_std:.2E}', 'mse': f'{mse:.2E} ± {mse_std:.2E}', 'mape': f'{mape:.2E} ± {mape_std:.2E}', 'params': group.iloc[0].params}
            data_dictionary.update(dict(zip(groupby_cols, list(grouped_variables))))
            
            metrics = pd.DataFrame(data_dictionary, index=[0])

            grouped_metrics = pd.concat((grouped_metrics, metrics))
    
    return grouped_metrics