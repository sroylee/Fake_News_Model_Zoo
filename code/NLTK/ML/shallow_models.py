"""
Caim Chen
This class has following functionality:
 1. Read in desired dataset(pre-prosessed)
 2. Taking models and evaluate Scores.
"""
import os
import pickle
import warnings

import nltk
import json
import numpy as np

import pandas as pd
from sklearn.model_selection import (StratifiedKFold)

# data resample
from sklearn.utils import resample
import imblearn as imbalan

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

import markdown_table as mt

# ----------------pay attention to following.
# np.random.seed(42)



path1 = 'D:/cmpt400-project/Fake_News_Model_Zoo/code/NLTK/datasets/binary/'
path2 = 'D:/cmpt400-project/Fake_News_Model_Zoo/code/NLTK/datasets/v1_dataset/'

def read_data(datatype):
    if datatype == 'binary':
        train_data = pd.read_csv(path1 + 'train.csv')
        valid_data = pd.read_csv(path1 + 'valid.csv')
        test_data = pd.read_csv(path1 + 'test.csv')
        return train_data,valid_data,test_data
    elif datatype == 'muti':
        train_data = pd.read_csv(path2 + 'train.csv')
        valid_data = pd.read_csv(path2 + 'valid.csv')
        test_data = pd.read_csv(path2 + 'test.csv')
        return train_data, valid_data, test_data
    else:
        return None

# Helper function to display the evaluation metrics of the different models

#show_eval_scores
def model_eva(datatype,model,model_name):
    """Function to show to different evaluation score of the model passed
    on the test set.

    Parameters:
    -----------
    model: scikit-learn object
        The model whose scores are to be shown.
    model_name: string
        The name of the model.
    """
    train_data, valid_data, test_data = read_data(datatype)
    if train_data.empty or valid_data.empty or test_data.empty is True:
        raise ValueError('Invalid dataset, check files')
    # Prepare the storage for data
    accu,prec,recall,f1 = ([] for i in range(4))
    col = ["Model Name", "Accuracy", "Precision", "Recall", "F1 score"]


    training_set = pd.concat([train_data, valid_data], ignore_index=True)
    # -----preview of sample from different data set.
    # print(train_data.sample(3))
    # print()
    # print(valid_data.sample(3))
    # print()
    # print(test_data.sample(3))
    # print()

    # nxn size of the data set.
    # print('Train dataset size: {}'.format(train_data.shape))
    # print('Valid dataset size: {}'.format(valid_data.shape))
    # print('Test dataset size: {}'.format(test_data.shape))
    # print('Training set size: {}'.format(training_set.shape))
    # print(training_set.sample(5))


    # using Kfold-cross validation
    # Each Model Error estimation calculated by cross validation.
    scross = StratifiedKFold(n_splits=5, random_state=777, shuffle=True)
    # Go through each fold then test it.
    for train_index,test_index in scross.split(training_set['news'],training_set['label']):

        learned = model.fit(training_set['news'][train_index],training_set['label'][train_index])
        # use trained model to predict test data
        y_pred = learned.predict(training_set['news'][test_index])
        y_true = training_set['label'][test_index]

        accu.append(accuracy_score(y_true, y_pred))
        prec.append(precision_score(y_true, y_pred,average = 'weighted'))
        recall.append(recall_score(y_true, y_pred,average = 'weighted'))
        f1.append(f1_score(y_true, y_pred,average = 'weighted'))

    matrix = [[model_name, "{0:.2f}".format(np.mean(accu)),
               "{0:.2f}".format(np.mean(prec)),
               "{0:.2f}".format(np.mean(recall)),
               "{0:.2f}".format(np.mean(f1))]]

    table = mt.Table(col, matrix)
    print("Result table for StratifiedKFold on dataset",datatype)
    print(str(table))
    #------------- Processed to real test set --------------
    print()
    print("Result table for real testing on dataset",datatype)
    test_model = model.fit(training_set['news'], training_set['label'])
    test_pred = test_model.predict(test_data['news'])
    test_label = test_data['label']

    matrix2 = [[model_name, "{0:.2f}".format(accuracy_score(test_label, test_pred)),
               "{0:.2f}".format(precision_score(test_label, test_pred,average = 'weighted')),
               "{0:.2f}".format(recall_score(test_label, test_pred,average = 'weighted')),
               "{0:.2f}".format(f1_score(test_label, test_pred,average = 'weighted'))]]

    table2 = mt.Table(col, matrix2)
    print(str(table2))

"""
Upsampling: Where you increase the frequency of the samples, such as from minutes to seconds.
Downsampling: Where you decrease the frequency of the samples, such as from days to months.
"""

''' Print the vocabulary occurrence '''
# print(CountVec.vocabulary_)
# print(TfidfVec.vocabulary_)
''' Print number of features '''
# print('Number of feature for CountVec:',len(CountVec.get_feature_names()))
# print('Number of feature for TfidfVec:',len(TfidfVec.get_feature_names()))