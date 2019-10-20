import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from .data_preprocessing import standardize_dataset

# foundation of reading data
def read_dataset(csv_file_path, feature_index, label_index, feature_start_index = 0,
                 training_data_scaling_function = None, data_scaler_path = None):
    '''
    csv file format:
    feature1, feature2,...,label
    '''
    df = pd.read_csv(csv_file_path)
    feature = df[df.columns[feature_start_index:feature_index]].values      #features
    if training_data_scaling_function is not None:
        feature = standardize_dataset(dataset=feature, scaler_path = data_scaler_path, standardization_scaler_function = training_data_scaling_function)

    label = df[df.columns[label_index]]                                     #labels

    encoder = LabelEncoder()
    encoder.fit(label)

    distinct_label = list(set(label))
    labelMap = encoder.transform(distinct_label)
    print("Label Mapping")
    print('\n'.join([str(distinct_label[i]) + ' -> ' + str(labelMap[i]) for i in range(len(distinct_label))]))
    label = encoder.transform(label)            # sign and transform class to a number 0, 1, 2 ...
    label_matrix = one_hot_encode(label)        # to be column_num * num_class_type matrix
    return (feature, label_matrix, label)

def one_hot_encode(labels):
    n_lables = len(labels)
    n_unique_lables = len(np.unique(labels))
    one_hot_encode = np.zeros((n_lables, n_unique_lables))
    one_hot_encode[np.arange(n_lables), labels] = 1
    return one_hot_encode
