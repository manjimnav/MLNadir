from pickle import load
from collections import deque
import numpy as np
from datetime import datetime
import argparse

if __name__== "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="The input file with 'AngB1_unwrapped', 'AngB2_unwrapped', 'AngB3_unwrapped' along time.")
    parser.add_argument("-o","--output", help="The output file where the predictions will be saved.")

    args = parser.parse_args()
    with open('models/best_model_100ms.joblib', 'rb') as fid:
        model = load(fid)
    
    features = 3
    window_size = model.n_features_in_//features

    input_model = np.genfromtxt(args.input,delimiter=';') 

    input_model = input_model.flatten('F')

    prediction = model.predict([input_model])[0]
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(args.output, 'a') as out_file:
        out_file.write(f'{date};{prediction}\n')