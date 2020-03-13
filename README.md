# Caim_Fake_News_Model_Zoo
Caim Chen's CMPT400 project. 
# Log for week Mar 7th

1.Explore attention model and bleu score, did not observe any noticeable improvements. ( Next step: try tensorflow.keras attention layer).

2.Keras attention layer followed encoder decoder format, could not adpat to fake news detection task.

3.Possible causes : Encoder and decoder are designed for transltor purpose, thismight be unsuitable to categorical classfication task.

4.Learned how to analysis Lair dataset from each feature provided in raw data.

5.Points learned : 1. using each feature to extract truthfulness relation with the labels.(bar chart) 2.using groupby to select extreme cases from two different features then combine them with heatmap. (ex.which party would have the highest lie rate, which job would have the highest lie rate, how about the combination feature?) 3. Finding biased properties before constructing the model. ( ex. number of republican is 6 times bigger than democratic) 

# Log for week Mar 1st

1.Redo everything for data pre-processing. Vocabulary coverage is 78% before any pre-process been done. ( cleaned symbol,punctuation,number,different spelling in each region. Possible for further mis-spelling cleaning )

2.Redo everything for embedding creation. Achieved average of 99% vocabulary coverage after applying all pre-processing.

3.Adjust optimizers' hyperparameters to achieve better performance.

4.Saving the best model for later uses.

5.CNN accuracy is now around 26%, BiLSTM accuracy is now around 22%. Matched the Liar result, able to proceed to the next step.

# Log for week Feb 17th 
Code for CNN can be found under Fake_News_Model_Zoo/code/DL/FakeNewsCNN.ipynb

1.Refactored directory layout.

2.Switch from glove.6B.50d.txt to GoogleNews-vectors-negative300.bin as a better embedding option. ( More reliable for news detection )

3.Adjusted parameters to match the paper description for CNN.

4.Tried different learning rates, Maxpooling, Averagepooling, optimizer and embedding's trainable option.

5.Able to achieve better accuracy on the testing set. accu : 19~20%, still cannot match 26% accuracy according to the paper description.

6.Changed to the unprocessed version of the data with punctuation removed, still unable to match the results. However, I was able to remove one error that caused validation set accuracy to reach 26%. ( Might missing something during setting up the input dataset )

7.After took the udemy online course on RNN and LSTM, followed the tutorial and implemented the LSTM that does prediction on the stock price.

8.Implement Bi-LSTM, paper has 22%-23% accuracy on validation and testing set. However, my implementation only yield an accuracy to 19%.

# Log for week Feb 10th
1.Clean up the shallow models' code, and updated structure. ( Now can choose a different version of data set to use, and more robust.)

2.Updated the summary structure in the README file.

3.Took online udemy course on CNN, implemented dogs and cats classification.

4.Follow [Text Classification Tutorial](https://realpython.com/python-keras-text-classification/), implemented CNN for fake news detection. CNN model performed ok on validation dataset, but it has clear overfitting results on testing set.

5.Validation dataset: accu from 20% ~ 25%,loss from 1.8 ~ 1.7. Testing dataset: accu 16%

6.Added in glove matrix based on (glove.6B.50d.txt), shows 93% coverage on the vocabulary.

7.Validation dataset: accu from 20% ~ 26%,loss from 1.8 ~ 1.7.Testing dataset: accu 16~18%

Conclusion(Diffculties):

1.Model learned too well and caused overfitting.

2.Article body pattern in train, valid and test dataset might be too distinct for model to understand and predict due to lack of background knowledge.

3.Unable to build a model that reduce bias and variance at same time.

# Shallow Models
### Datasets
1. [Fact-Checking Facebook Politics](https://arxiv.org/pdf/1705.00648.pdf): LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION(Include politic fact, political debate, TV ads, Facebook posts, tweets, interview, news release, etc.)
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

[wcbarely-true]:Figure/wcbarely-true.png "true"
[wcfalse]:Figure/wcfalse.png "mostly-true"
[wchalf-true]:Figure/wchalf-true.png "half-true "
[wcmostly-true]:Figure/wcmostly-true.png "false"
[wcpants-fire]:Figure/wcpants-fire.png "barely-true"
[wctrue]:Figure/wctrue.png "pants-fire"


1.True Word Cloud
![alt text][wctrue]

---

2.Mostly-true Word Cloud
![alt text][wcmostly-true]

---

3.Half-true Word Cloud
![alt text][wchalf-true]

---

4.False Word Cloud
![alt text][wcfalse]

---

5.Barely-true Word Cloud
![alt text][wcbarely-true]

---

6.Pants-fire Word Cloud
![alt text][wcpants-fire]


---
#### Data Pre-Processing Steps
1. Read in "Lair" dataset( train.tsv | valid.tsv | test.tsv ).

2. Select columns #1 and #2 in the three input dataset, use label and news corresponding to #1 and #2 as the new columns' name.

3. Map multiclass labels to binary class labels for the binary analyzing case. Muti-label analyzes can directly go to Data Pre-processing Steps 6.

4. After taking consider of binary labels, perform the same procedure on the original multi-label dataset
Reference table as follow:

| Multiclass labels  | Binary labels|
| ------------- |:-------------:|
| true   | true |
| mostly-true   | true |
| half-true   | true |
| false   | false |
| barely-true   | false |
| pants-fire   | false |

4. Convert all string content in the 'news' category into lower case string.

5. Check for dataset quality before creating .csv file ( Remove the entry contains missing values ).

6. Using StratifiedKFold methods to compute the estimate confidence level for real-world data.

7. A data pre-processing workflow helper function is built to organize all the above steps into one function call.

##### Reason for pre-processing

1.Clean the dataset by removing factors not under consideration in the current stage. (such as subject title,speaker job.)

2.Improve the dataset quality by removing invalid entries.

3.With binary labels, easier to do classification.

4.Increased the sample size by using binary labelled classification, give each model more data to work with. 

---

### Traditional Machine Learning Methods
1.Logistic Regression.

2.Naive Bayes.

3.Supporter Vector Machine.

4.Decision Tree.

5.Random Forest.

---
### Data process steps
#### Prepare the martix for models
1.Read dataset files.

2.Concate training and valid datasets.

3.Convert a collection of text documents to a matrix of token counts.

4.After analyzing the word cloud for 6 categories(true,half-true...false), word frequencies entails certain pattern for the data. Perform a TF-IDF transform to the obtained word count matrix, the result after transform is in normalized tf-idf representation matrix.

---
#### Improve the matrix
1.Using n-gram(1,2) generally,since it yield better result compare to (1,1)(1,2)(1,3)(2,2)(3,3). Due the average words appear in each news article is around 20, considering beyond tri-gram will bring biased results. 

2.Removing common stop words by using built-in stop word list for English. Clearing stop words to eliminate distracting factors.

3.In the context of text classification, take in count of char n-gram would be unnecessary. Setting analyzer string to word to eliminate distracting factors.

#### Input processed matrix to corresponding models
---
##### Parameters changes when implementing traditional machine learning methods
1.Logistic Regression.

  + Set random_state to 42, to yield the same results for testing purpose.
  + Set n_jobs to -1,to use all processors which increase the speed of calculation.

2.Decision Tree
  + Change ngram_range to (1,3) due to better precision and f1-score.
  + Use random state 42 to fix the result of splitting the tree.

3.Random Forest
  + Change ngram_range to (1,3) due to better precision and f1-score.
  + Use random state 42 to fix the result of splitting the tree.
  + Due to long computation time, reduce max_depth to 20 which achieves faster computation.
  + Since max_depth is reduced, using n_estimators = 500 to slightly bringing more trees in to consideration. More chances to find a better DT.
  + Set n_jobs to -1,to use all processors which increase the speed of calculation.
##### Reasons for using specific type of models

1.Multinomial Naive Bayes.

  + Due to bag of words bringing large number of features, using multinomial NB rather than Bernoulli NB.
  
2.Complement Naive Bayes

  + Considering the proportion of imbalanced dataset, use complement NB to compare with multinomial.
  
---
#### Results

1. Binary Labeled Result
  + StratifiedKFold
  
Model Name|Accuracy|Precision|Recall|F1 score
-|-|-|-|-
logistic regression |0.62 |0.61 |0.62 |0.60
Multinomial Naive Bayes |0.60 |0.62 |0.60 |0.55
Complement Naive Bayes |0.25 |0.25 |0.25 |0.24
Support Vector Machine |0.61 |0.60 |0.61 |0.60
Decision Tree |0.55 |0.55 |0.55 |0.55
Random Forest |0.57 |0.64 |0.57 |0.43

  + Test
  
Model Name|Accuracy|Precision|Recall|F1 score
-|-|-|-|-
logistic regression |0.62 |0.62 |0.62 |0.61
Multinomial Naive Bayes |0.62 |0.65 |0.62 |0.56
Complement Naive Bayes |0.25 |0.25 |0.25 |0.25
Support Vector Machine |0.62 |0.62 |0.62 |0.62
Decision Tree |0.52 |0.52 |0.52 |0.52
Random Forest |0.58 |0.70 |0.58 |0.44

---
2. Muti Labeled Result
  + StratifiedKFold
  
Model Name|Accuracy|Precision|Recall|F1 score
-|-|-|-|-
logistic regression |0.25 |0.28 |0.25 |0.24
Multinomial Naive Bayes |0.24 |0.23 |0.24 |0.19
Complement Naive Bayes |0.25 |0.25 |0.25 |0.24
Support Vector Machine |0.24 |0.24 |0.24 |0.24
Decision Tree |0.21 |0.20 |0.21 |0.20
Random Forest |0.22 |0.17 |0.22 |0.14
  + Test
  
Model Name|Accuracy|Precision|Recall|F1 score
-|-|-|-|-
logistic regression |0.25 |0.26 |0.25 |0.24
Multinomial Naive Bayes |0.23 |0.22 |0.23 |0.19
Complement Naive Bayes |0.25 |0.25 |0.25 |0.25
Support Vector Machine |0.25 |0.25 |0.25 |0.25
Decision Tree |0.21 |0.20 |0.21 |0.20
Random Forest |0.21 |0.13 |0.21 |0.13

---
#### Discussion
1.ComplementNB yield better score than MultinomialNB, which proved imbalanced property does exist in the used dataset.

2.Random Forest has better F1 score and Recall score than Decision Tree, caused by more connection between features got examined.

3.MultinomialNB holds the highest F1 score, entails MultinomialNB is most stable model to describe the data content.

---

#### Further work
1.Considering part of speech, words belongs to noun should receive better weights by using Universal Part-of-Speech Tagset.

2.Finding new features like length of article or article subject title to increase prediction score. 

3.Continue with generating visual graph of decision tree, by doing so will give clues about how to adjust max_features and bootstrap.

4.Having the classification report that describe Micro&Macro average, come up with better solution to fix an imbalanced dataset.
