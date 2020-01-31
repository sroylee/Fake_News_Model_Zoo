# Caim_Fake_News_Model_Zoo
Caim Chen's CMPT400 project. 

# Shallow Models
### Datasets
1. [Fact-Checking Facebook Politics](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip): LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION(Include politic fact, political debate, TV ads, Facebook posts, tweets, interview, news release, etc.)
2. [Fake-News-Detection-System](https://github.com/raj1603chdry/Fake-News-Detection-System/tree/master/datasets): Preprocessed Liar dataset(From `multiclass labels` to `binary labels`, detailed process included in Data Pre-Processing section)

---
#### Dataset Table
1.Liar Dataset

| Type of usage   | #True Label   | #Mostly-true Label   | #Half-true Label| #False Label| #Barely-true Label |#Pants-fire Label|News Content |
| ------------- |----|----|----|----|----|----| -----:|
| Train|   3355 |  1966  | 2123   |  2011 |  1666   |  849 | politifact |
| Valid| 411   | 252   |  248  | 265  |  238   | 116  |politifact |
| Test | 424   | 249   |  270  |  252 | 214    | 93  |politifact |


2.Preprocessed Liar Dataset

| Type of usage | #True Label   | #False Label   | News Content |
| ------------- |----|----| -----:|
| Train| 5752    | 4488   | politifact |
| Valid|  669    | 617   |   politifact |
| Test |  714   | 553   |    politifact |

---
#### Dataset Chart
[o_train]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/o_train.png "Trainning Dataset"
[o_test]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/o_test.png "Testing Dataset"
[o_valid]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/o_valid.png "Valid Dataset"

[train]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/train.png "Trainning Dataset"
[test]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/test.png "Testing Dataset"
[valid]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/Figure/valid.png "Valid Dataset"

1.Liar Dataset
![alt text][o_train]


![alt text][o_test]


![alt text][o_valid]

2.Preprocessed Liar Dataset
![alt text][train]


![alt text][test]


![alt text][valid]

---
#### Dataset Wordcloud
[wc_train]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/code/NLTK/graph/wc_train.png "Trainning Dataset Wordcloud"
[wc_valid]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/code/NLTK/graph/wc_valid.png "Validing Dataset Wordcloud"
[wc_test]: https://github.com/sroylee/Fake_News_Model_Zoo/blob/master/code/NLTK/graph/wc_test.png "Testing Dataset Wordcloud"

1.Trainning Dataset Word Cloud
![alt text][wc_train]

2.Validing Dataset Word Cloud
![alt text][wc_valid]

3.Testing Dataset Word Cloud
![alt text][wc_test]

---
#### Data Pre-Processing Steps
1. Read in "Lair" dataset( train.tsv | valid.tsv | test.tsv ).

2. Select columns #1 and #2 in the three input dataset, use label and news corrsponding to #1 and #2 as the new columns' name.

3. Map multicalss labels to binary class labels.

Reference table as follow:

| Multiclass labels  | Binary labels|
| ------------- |:-------------:|
| true   | true |
| mostly-true   | true |
| half-true   | true |
| false   | false |
| barely-true   | false |
| pants-fire   | false |

4. Convert all string content in 'news' category in to lower case string.

5. Check for dataset quality before creating .csv file ( Remove the entry contains missing values ).

##### Reason for pre-processing

1.Clean the dataset by removing factors not under consideration in current stage.(such as subject title,speaker job.)

2.Improve the dataset quality by removing invalid entries.

3.With binary labels, easier to do classfication.

4.Increased the sample size by using binary labeled classfication, give each model more data to work with. 

---

### Traditional Machine Learning Methods
1.Logistic Regression.

2.Naive Bayes.

3.Suppoter Vector Machine.

4.Decision Tree.

5.Random Forest.

---
### Data process steps
#### Prepare the martix for models
1.Read dataset files.

2.Concate training and valid datasets.

3.Convert a collection of text documents to a matrix of token counts

4.Transform a count matrix to a normalized tf or tf-idf representation

---
#### Improve the martix
1.Using ngram(1,2) generally,since it yeild better result compare to (1,1)(1,2)(1,3)(2,2)(3,3). Due the average words appear in each news article is around 20, considering beyond tri-gram will bring biased results. 

2.Removing common stop words by using built-in stop word list for English. Clearing stop words to eliminate distracting factorseliminate distracting factors.

3.In the context of text classfication, take in count of char-ngram would be unecessary. Setting analyzer string to word to eliminate distracting factors.

#### Input processed matrix to corresponding models
---
##### Parameters changes when implementing traditional machine learning methods
1.Logistic Regression.

  + Set random_state to 42, to yeild same results for tesing purpose.
  + Set n_jobs to -1,to use all processor which increase the speed of calculation.

2.Decision Tree
  + Change ngram_range to (1,3) due to better precision and f1-score.
  + Use random state 42 to fix the result of splitting the tree.

3.Random Forest
  + Change ngram_range to (1,3) due to better precision and f1-score.
  + Use random state 42 to fix the result of splitting the tree.
  + Due to long computation time, reduce max_depth to 20 which achieves faster computation.
  + Since max_depth is reduced, using n_estimators = 500 to slightly bringing more trees in to consideration. More chances to find a better DT.
  + Set n_jobs to -1,to use all processor which increase the speed of calculation.
##### Reasons for using specfic type of models

1.Multinomial Naive Bayes.

  + Due to bag of words brining large number of features, using multinomial NB rather than bernoulli NB.
  
2.Complement Naive Bayes

  + Considering the propotion of imbalanced dataset, use complement NB to compare with multinomial.
  
---
#### Results

1. Binary Labeled Result

| Model Name  | Accuracy  | Precision score   |Recall score| F1 score|
| ------- |----|----|----|----:|
| Logistic Regression|   0.6237308602999211 |  0.6322869955156951    |  0.7899159663865546  | 0.7023661270236612|
| MultinomialNB|   0.6203630623520127 |  0.6077705827937095     |  0.9201680672268907 | 0.732033426183844|
| ComplementNB|  0.6203630623520127  |  0.6230200633579726    | 0.8263305322128851  |  0.7104154124021673|
| Support Vector Machine|   0.6211523283346487 |  0.6451612903225806   |0.7282913165266106   |   0.6842105263157894|
| Decision Tree|  0.5177584846093133  |  0.5791090629800307    | 0.5280112044817927  |  0.5523809523809524|
| Random Forest| 0.5753749013417522   |   0.5706260032102729   |  0.9957983193277311 |  0.7255102040816327|

2. Muti Labeled Result

+ Logistic Regression
![alt text][lr]

+ MultinomialNB
![alt text][mnb]

+ ComplementNB
![alt text][cnb]

+ Support Vector Machine
![alt text][svm]

+ Decision Tree
![alt text][dt]

+ Random Forest
![alt text][rf]

---
#### Discussion
1.ComplementNB yeild better score than MultinomialNB, which proved imbalanced property does exist in the used dataset.

2.Random Forest havs better F1 score and Recall score than Decision Tree, caused by more connection between features got examined.

3.MultinomialNB holds the highest F1 score, entails MultinomialNB is most stable model to describe the data content.

---

#### Further work
1.Considering part of speech, words belongs to noun should recieve better weights by using Universal Part-of-Speech Tagset.

2.Finding new features like length of article or article subject title to increase prediction score. 

3.Continue with generating visual graph of decision tree, by doing so will give clues about how to adjust max_features and bootstrap.

4.Having the classfication report that describe Micro&Macro average, come up with better solution to fix imbalanced dataset.
