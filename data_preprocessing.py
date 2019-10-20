from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import math

from sklearn import preprocessing
from sklearn.externals import joblib

def standardize_csv_data(input_file_path, output_file_path, begin_column_num, end_column_num, has_header = None, index = None, scaler_path=None):
    no = end_column_num - begin_column_num
    df = pd.read_csv(input_file_path,  index_col=False, header = None, names = [i for i in range(no)])
    df = df[df.columns[begin_column_num:end_column_num]].values
    standard_df = standardize_dataset(df, scaler_path)
    #standard_df.columns = df.columns
    #standard_df.reset_index(df.index.values)
    pd.DataFrame(standard_df).to_csv(output_file_path)


def standardize_dataset(dataset, scaler_path=None, standardization_scaler_function = preprocessing.StandardScaler()):
    scaler = standardization_scaler_function.fit(dataset)

    # store the standard scaler into scaler_path
    if scaler_path is not None:
        joblib.dump(scaler, scaler_path)
        
    scaled_data = scaler.transform(dataset)

    if type(dataset) == pd.DataFrame:
        scaled_data = pd.DataFrame(scaled_data, index=dataset.index, columns=dataset.columns)

    return scaled_data

if __name__ == '__main__':
    df = pd.read_csv('../resource/sonar.all-data.csv', index_col=False,
                     header = None, names = [i for i in range(61)])
    standardize_csv_data('../resource/sonar.all-data.csv', '../resource/sonar.all-data_standardized.csv', 0, 60)

    df = pd.read_csv('../resource/sonar.all-data.csv', index_col=False,
                     header = None, names = [i for i in range(61)])
    dataset = df.loc[:, 0:59]
    dataset1 = standardize_dataset(dataset, './filename.pkl')
    scaler = joblib.load('./filename.pkl')
    dataset2 = pd.DataFrame(scaler.transform(dataset) , index=dataset.index, columns=dataset.columns)

    print(dataset1.equals(dataset2))
