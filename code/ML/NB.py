from sklearn.feature_extraction.text import (CountVectorizer,TfidfTransformer)
from sklearn.naive_bayes import (MultinomialNB,ComplementNB)
from sklearn import pipeline

from nltk.corpus import stopwords

from shallow_models import model_eva as me

stopwords_list = list(stopwords.words('english'))

print('------------------------------------------------------------------------')
print('Naive Bayes ')

# Multinomial Naive Bayes

nb_pipe_multi_tfidf = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_Muti', MultinomialNB())
])
me("binary",nb_pipe_multi_tfidf,"Multinomial Naive Bayes")
# ++++++++++++++++++++++++++++++

print()
# Complement Naive Bayes

nb_pipe_Com = pipeline.Pipeline([
    ('CV', CountVectorizer(stop_words=stopwords_list, lowercase=False,ngram_range=(1,2),analyzer = 'word')),
    ('TFIDF-Trans', TfidfTransformer()),
    ('nb_comple', ComplementNB())
])

me("muti",nb_pipe_Com,"Complement Naive Bayes")