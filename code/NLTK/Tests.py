
#
# from sklearn.feature_extraction.text import CountVectorizer
# # list of text documents
# text = ["The quick brown fox jumped over the lazy puppy."]
# # create the transform
# vectorizer = CountVectorizer()
# # tokenize and build vocab
# vectorizer.fit(text)
# # summarize
# print(vectorizer.vocabulary_)
# # encode document
# vector = vectorizer.transform(text)
# # summarize encoded vector
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
#
# # encode another document
# text2 = ["the puppy"]
# vector = vectorizer.transform(text2)
# print(vector.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
# vector = vectorizer.transform([text[1]])
# # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())
# vector = vectorizer.transform([text[2]])
# # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())