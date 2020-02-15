from sklearn.feature_extraction.text import (CountVectorizer,TfidfVectorizer,TfidfTransformer)
from sklearn import pipeline
from nltk.corpus import stopwords
from sklearn import svm
from shallow_models import model_eva as me

stopwords_list = list(stopwords.words('english'))
# SVM
print('------------------------------------------------------------------------')
print('Support Vector Machine')
SVM_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('svm', svm.LinearSVC())
])
me("binary",SVM_tfidf,"Support Vector Machine")