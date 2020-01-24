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
from sklearn.tree import (DecisionTreeClassifier,export_text)
from sklearn import tree


from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
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
print(stopwords_list)


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
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('Model Name: {}'.format(model_name))
    print('Accuracy: {}'.format(accuracy))
    print('Precision score: {}'.format(precision))
    print('F1 score: {}'.format(f1))
    print('Recall score: {}'.format(recall))

train_data = pd.read_csv('./datasets/binary/train.csv')
valid_data = pd.read_csv('./datasets/binary/valid.csv')
test_data = pd.read_csv('./datasets/binary/test.csv')

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
print(training_set.sample(5))

# Two type vectorizer
CountVec = CountVectorizer()
TfidfVec = TfidfVectorizer()

train_count1 = CountVec.fit_transform(training_set['news'].values)

train_count2 = TfidfVec.fit_transform(training_set['news'].values)

''' Print the vocabulary occurrence '''
# print(CountVec.vocabulary_)
# print(TfidfVec.vocabulary_)
''' Print number of features '''
# print('Number of feature for CountVec:',len(CountVec.get_feature_names()))
# print('Number of feature for TfidfVec:',len(TfidfVec.get_feature_names()))


print('------------------------------------------------------------------------')
print('logistic regression')

# # Using CountVec
# lr_pipe_CV = Pipeline([
#     ('LR-CV', CountVectorizer(stop_words=stopwords_list, lowercase=False)),
#     ('LR', LogisticRegression(C=0.0001,random_state=42, n_jobs=-1))
# ])
#
# lr_pipe_CV.fit(training_set['news'], training_set['label'])
# show_eval_scores(lr_pipe_CV, test_data, 'Logistic Regression-CV')

# print()
# Using TfidfVec + CountVec

lr_tfidf = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    ('LR', LogisticRegression(C=0.0001,random_state=42, n_jobs=-1))
])

lr_tfidf.fit(training_set['news'], training_set['label'])
show_eval_scores(lr_tfidf, test_data, 'Logistic Regression-CV-TFIDF')


print('------------------------------------------------------------------------')
print('Naive Bayes ')

# Multinomial Naive Bayes

nb_pipe_multi_tfidf = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_Muti', MultinomialNB(alpha=6.8))
])
nb_pipe_multi_tfidf.fit(training_set['news'], training_set['label'])
show_eval_scores(nb_pipe_multi_tfidf, test_data, 'MultinomialNB-CV-TFIDF')

print()
# Complement Naive Bayes

''' Since the dataset is '''
nb_pipe_Com = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_comple', ComplementNB(alpha=6.8))
])
nb_pipe_Com.fit(training_set['news'], training_set['label'])
show_eval_scores(nb_pipe_Com, test_data, 'ComplementNB-CV-TFIDF')

# SVM
print('------------------------------------------------------------------------')
print('Support Vector Machine')
SVM_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    ('svm', svm.LinearSVC())
])

SVM_tfidf.fit(training_set['news'], training_set['label'])
show_eval_scores(SVM_tfidf, test_data, 'SVM-CV-TFIDF')

print()

'''
Use random state 42 to fix the result of splitting the tree.
'''

print('------------------------------------------------------------------------')
print('Decision Tree')

DT_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    ('DT', DecisionTreeClassifier(random_state=42))
])

DT_tfidf.fit(training_set['news'], training_set['label'])
show_eval_scores(DT_tfidf, test_data, 'DT-CV-TFIDF')

# r = export_text(DT_tfidf)
# print(r)
# with open("./graph/news.dot","w") as f:
#     dot_data = tree.export_graphviz(DT_tfidf.named_steps['DT'],out_file=f)
# graph = graphviz.Source(dot_data)
# graph.render("what")


print('------------------------------------------------------------------------')
print('Random Forest')

RF_TFIDF = Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(3,3))),
    ('TFIDF-Trans', TfidfTransformer()),
    # ('RF', RandomForestClassifier(max_depth=12, n_estimators=300, n_jobs=-1, random_state=42))
    ('RF', RandomForestClassifier(random_state=42))
])
RF_TFIDF.fit(training_set['news'], training_set['label'])
show_eval_scores(RF_TFIDF, test_data, 'Random Forest')