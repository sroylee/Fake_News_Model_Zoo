from sklearn.feature_extraction.text import (CountVectorizer,TfidfTransformer)
from sklearn import pipeline
from nltk.corpus import stopwords
from shallow_models import model_eva as me
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


stopwords_list = list(stopwords.words('english'))

print('------------------------------------------------------------------------')
print('Decision Tree')

DT_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,3),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('DT', DecisionTreeClassifier(random_state=42))
])
me("muti",DT_tfidf,"Decision Tree")

print('------------------------------------------------------------------------')
print('Random Forest')

RF_TFIDF = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,3),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('RF', RandomForestClassifier(max_depth=20,n_estimators=500, n_jobs=-1, random_state=42))
])
me("muti",RF_TFIDF,"Random Forest")