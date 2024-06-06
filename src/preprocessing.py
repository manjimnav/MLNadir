import pandas as pd
from tqdm.notebook import tqdm
import os
import re

def generate_code(metadata):
    """
    Generates unique code for identify every study
    """
    code = ''
    if len(metadata)==3:
        code = f'{metadata[0]}_{metadata[1]}_{metadata[2]}'
    else:
        code = f'{metadata[0]}_{metadata[1]}_{metadata[2]}_{metadata[3]}'
    
    return code

def add_metadata(df, file):
    regexPattern = r"(\d+)"
    expression = file
    metadata = re.findall(regexPattern,expression)

    df['scenario'] = f'{metadata[0]}'
    df['case'] = f'{metadata[1]}'

    if len(metadata)>3:
        df['subcase'] = f'{metadata[2]}'
        df['result'] = f'{metadata[3]}'
    else:
        df['result'] = f'{metadata[2]}'
    
    code = generate_code(metadata)
    df['code'] = code
        
    return df

def get_target(df, folder):
    if 'DisparoGenerador' in folder:
        target = df['Fmin (Hz)'].min()
    else:
        target = df['Fmax (Hz)'].max()
    
    return target
        
def generate_dataset(folders):
    data = pd.DataFrame()
    for folder in tqdm(folders):
        for file in tqdm(os.listdir(f'data/raw/{folder}')):
            
            df = pd.read_excel(f'data/raw/{folder}/{file}')
            
            #Select input data from sec 1 to 3
            shortened_data = df.loc[(df['Time (s)']<=3) & (df['Time (s)']>=1)]
            
            shortened_data = shortened_data.set_index('Time (s)')

            shortened_data['target'] = get_target(df, folder) # Add the global target (note we use df instead of shortened_data) depending if the goal is to predict the global maximum of minimum
            shortened_data = add_metadata(shortened_data, file) # Add metadata for posterior analysis
            
            # Concatenate data into one dataframe
            data = pd.concat((data, shortened_data))
            
    return data


def windowing(data, columns, by=['code']):
     
    lagged_data = pd.DataFrame()

    for code, group in tqdm(data.groupby(by)):
        
        for interval in range(int(max(group.interval.tolist())+1)):
            row = group[columns][(group.interval==interval)]

            lagged_columns = [f'{col}-{lag}' for col in columns for lag in range(len(row)-1, -1, -1)]
            df_row = pd.DataFrame(data=[row.values.flatten('F')], columns=lagged_columns, index=[0])

            df_row['code'] = code
            df_row['interval'] = interval
            df_row['target'] = group.iloc[0]['target']
            df_row['pred_time'] =  group[(group.interval==interval)].iloc[-1]['Time (s)']

            lagged_data = pd.concat((lagged_data, df_row))
    return lagged_data