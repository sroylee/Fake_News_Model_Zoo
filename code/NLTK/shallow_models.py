import os
import pickle
import warnings

import nltk
import json
import numpy as np
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     learning_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn import (
    datasets, feature_extraction, model_selection, pipeline,
    svm, metrics
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')
#test_data = pd.read_csv('test.csv',index_col=1)
test_data = pd.read_csv('test.csv')

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

#combining into one
training_set = pd.concat([train_data, valid_data], ignore_index=True)
print('Training set size: {}'.format(training_set.shape))
print(training_set.sample(5))

countV = CountVectorizer()
train_count = countV.fit_transform(training_set['news'].values)

print(countV.vocabulary_)
print('Number of feature:',len(countV.get_feature_names()))

# term frequency–inverse document frequency

print('------------------------------------------------------------------------')
# logistic regression-CountVectorizer
print('logistic regression')
print('Result of processed pipeline')

lr_pipeline = Pipeline([
    ('lrCV', CountVectorizer(stop_words=stopwords_list, lowercase=False, ngram_range=(1, 2))),
    # ('lrCV', feature_extraction.text.CountVectorizer()),
    ('lr_clf', LogisticRegression(C=0.0001,random_state=42, n_jobs=-1))
])

lr_pipeline.fit(training_set['news'], training_set['label'])
show_eval_scores(lr_pipeline, test_data, 'Logistic Regression Count Vectorizer')

# Naive Bayes
print('------------------------------------------------------------------------')
print('Naive Bayes ')
print('Result of Naive Bayes ')

nb_pipeline = Pipeline([
    ('nb_CV', CountVectorizer(stop_words=stopwords_list, lowercase=False, ngram_range=(1, 2))),
    ('nb_clf', MultinomialNB(alpha=6.8))
])
nb_pipeline.fit(training_set['news'], training_set['label'])
show_eval_scores(nb_pipeline, test_data, 'Naive Bayes')

# SVM
print('------------------------------------------------------------------------')
print('SVM-high accuracy')
print('Result of SVM')
SVM = pipeline.Pipeline([
    ('counts', CountVectorizer(stop_words=stopwords_list, lowercase=False, ngram_range=(1, 2))),
    # ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('svm', svm.LinearSVC())
])
# SVM_matic = SVM.fit_transform(training_set['news'], training_set['label'])
SVM.fit(training_set['news'], training_set['label'])
show_eval_scores(nb_pipeline, test_data, 'SVM')

# random forest
# print('------------------------------------------------------------------------')
# print('random forest')
# print('Result of random forest')
#
# rf_pipeline = Pipeline([
#     ('rf_CV', CountVectorizer(stop_words=stopwords_list, lowercase=False, ngram_range=(1, 1))),
#     ('rf_clf', RandomForestClassifier(max_depth=12, n_estimators=300, n_jobs=-1, random_state=42))
# ])
# rf_pipeline.fit(training_set['news'], training_set['label'])
# show_eval_scores(rf_pipeline, test_data, 'Random Forest Classifier Count Vectorizer')