import os
import pickle
import warnings

import nltk
import json
import numpy as np
# nltk.download('stopwords')
import pandas as pd
import graphviz
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer,TfidfVectorizer,TfidfTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold,StratifiedShuffleSplit,cross_val_score)
from sklearn.base import clone as skclone
from sklearn.tree import (DecisionTreeClassifier,export_text)

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score,classification_report)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     learning_curve)
from sklearn.naive_bayes import (MultinomialNB,ComplementNB)
from sklearn.pipeline import Pipeline

from sklearn import (
    datasets, feature_extraction, model_selection, pipeline,
    svm, metrics
)

from sklearn.svm import SVC


# ----------------pay attention to following.
# np.random.seed(42)

stopwords_list = list(stopwords.words('english'))
# print(stopwords_list)


# Helper function to display the evaluation metrics of the different models
def show_eval_scores(model, test_set, model_name):
    """Function to show to different evaluation score of the model passed
    on the test set.

    Parameters:
    -----------
    model: scikit-learn object
        The model whose scores are to be shown.
    test_set: pandas dataframe
        The dataset on which the score of the model is to be shown.
    model_name: string
        The name of the model.
    """
    y_pred = model.predict(test_set['news'])
    y_true = test_set['label']
    f1 = f1_score(y_true, y_pred,average = 'weighted')
    precision = precision_score(y_true, y_pred,average = 'weighted')
    recall = recall_score(y_true, y_pred,average = 'weighted')
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true,y_pred)

    print('Model Name: {}'.format(model_name))
    print('Accuracy: {}'.format(accuracy))
    print('Precision score: {}'.format(precision))
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print()
    print(report)

path1 = './datasets/binary/'
path2 = './datasets/v1_dataset/'
train_data = pd.read_csv(path2 + 'train.csv')
valid_data = pd.read_csv(path2 + 'valid.csv')
test_data = pd.read_csv(path2 + 'test.csv')

#-----preview of sample from different data set.
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

'''
Clear the existing index and reset it in the result by setting the ignore_index option to True.
'''
training_set = pd.concat([train_data, valid_data], ignore_index=True)
print('Training set size: {}'.format(training_set.shape))
'''
Generate random samples from the fitted Gaussian distribution.
5 fold cross validation will be used for hyperparameter tuning the different models
'''
# print(training_set.sample(5))



# # Two type vectorizer
# CountVec = CountVectorizer()
# TfidfVec = TfidfVectorizer()
#
# train_count1 = CountVec.fit_transform(training_set['news'].values)
#
# train_count2 = TfidfVec.fit_transform(training_set['news'].values)

''' Print the vocabulary occurrence '''
# print(CountVec.vocabulary_)
# print(TfidfVec.vocabulary_)
''' Print number of features '''
# print('Number of feature for CountVec:',len(CountVec.get_feature_names()))
# print('Number of feature for TfidfVec:',len(TfidfVec.get_feature_names()))

# using Kfold-cross validation
# Each Model Error estimation calculated by cross validation.
scross = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
# print(scross.get_n_splits(test_data['news'],test_data['label']))
# for train_index, test_index in scross.split(test_data['news'], test_data['label']):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = test_data['news'][train_index], test_data['news'][test_index]
#     y_train, y_test = test_data['label'][train_index], test_data['label'][test_index]


print('------------------------------------------------------------------------')
print('logistic regression')

# Using TfidfVec + CountVec

lr_tfidf = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('LR', LogisticRegression(random_state=42, n_jobs=-1,max_iter=1000))
])
lr_cv = skclone(lr_tfidf)
lr_tfidf.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(lr_tfidf, test_data, 'Logistic Regression-CV-TFIDF')
# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(lr_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for LR:",scores)
print("StratifiedKFold mean score for LR:",scores.mean())


print('------------------------------------------------------------------------')
print('Naive Bayes ')

# Multinomial Naive Bayes

nb_pipe_multi_tfidf = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_Muti', MultinomialNB())
])
nb_muti_cv = skclone(nb_pipe_multi_tfidf)
nb_pipe_multi_tfidf.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(nb_pipe_multi_tfidf, test_data, 'MultinomialNB-CV-TFIDF')

# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(nb_muti_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for nb_muti_cv:",scores)
print("StratifiedKFold mean score for nb_muti_cv:",scores.mean())


print()
# Complement Naive Bayes

''' Since the dataset is imbalance '''
nb_pipe_Com = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_comple', ComplementNB())
])
nb_com_cv = skclone(nb_pipe_Com)
nb_pipe_Com.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(nb_pipe_Com, test_data, 'ComplementNB-CV-TFIDF')

# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(nb_com_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for NB_comple:",scores)
print("StratifiedKFold mean score for NB_comple:",scores.mean())

# SVM
print('------------------------------------------------------------------------')
print('Support Vector Machine')
SVM_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('svm', svm.LinearSVC())
])
SVM_cv = skclone(SVM_tfidf)
SVM_tfidf.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(SVM_tfidf, test_data, 'SVM-CV-TFIDF')
# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(SVM_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for SVM:",scores)
print("StratifiedKFold mean score for SVM:",scores.mean())

print()

'''
Use random state 42 to fix the result of splitting the tree.
'''

print('------------------------------------------------------------------------')
print('Decision Tree')

DT_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,3),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('DT', DecisionTreeClassifier(random_state=42))
])
DT_cv = skclone(DT_tfidf)
DT_tfidf.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(DT_tfidf, test_data, 'DT-CV-TFIDF')

# r = export_text(DT_tfidf)
# print(r)
# with open("./graph/news.dot","w") as f:
#     dot_data = tree.export_graphviz(DT_tfidf.named_steps['DT'],out_file=f)
# graph = graphviz.Source(dot_data)
# graph.render("what")
# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(DT_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for DT:",scores)
print("StratifiedKFold mean score for DT:",scores.mean())
print()

print('------------------------------------------------------------------------')
print('Random Forest')

RF_TFIDF = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,3),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('RF', RandomForestClassifier(max_depth=20,n_estimators=500, n_jobs=-1, random_state=42))
])
RF_cv = skclone(RF_TFIDF)
RF_TFIDF.fit(training_set['news'], training_set['label'])
print("Classification Report :")
show_eval_scores(RF_TFIDF, test_data, 'Random Forest')

# ++++++++++++++++++++++++++++++
print()
scores = cross_val_score(RF_cv, training_set['news'],training_set['label'],cv=scross)
print("StratifiedKFold score for RF:",scores)
print("StratifiedKFold mean score for RF:",scores.mean())
