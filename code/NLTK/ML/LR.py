from sklearn.feature_extraction.text import (CountVectorizer,TfidfTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn import pipeline
from nltk.corpus import stopwords

from shallow_models import model_eva as me

stopwords_list = list(stopwords.words('english'))

print('------------------------------------------------------------------------')
print('logistic regression')


#Using TfidfVec + CountVec

lr_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('LR', LogisticRegression(random_state=42, n_jobs=-1,max_iter=1000))
])

me("muti",lr_tfidf,"logistic regression")