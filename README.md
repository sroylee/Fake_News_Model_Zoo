# Caim_Fake_News_Model_Zoo
Caim Chen's CMPT400 project. 


### Datasets
1. [Fact-Checking Facebook Politics](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip): Lair dataset
2. [Fake-News-Detection-System](https://github.com/raj1603chdry/Fake-News-Detection-System/tree/master/datasets): Preprocessed Lair dataset

| Dataset        | #News        | Classes  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

#### Data Pre-Processing
1. Read in "Lair" dataset( train.tsv | valid.tsv | test.tsv ).

2. Select columns #1 and #2 in the three input dataset, use label and news corrsponding to #1 and #2 as the new columns' name.

3. Map multicalss labels to binary class labels.

Reference table as follow:

| Multicalss labels  | Binary labels|
| ------------- |:-------------:|
| true   | true |
| mostly-true   | true |
| half-true   | true |
| false   | false |
| barely-true   | false |
| pants-fire   | false |

4. Convert all string content in 'news' category in to lower case string.

5. Check for dataset quality before creating .csv file ( Remove the entry contains missing values ).

### Traditional Methods
Logistic Regression
Naive Bayes 
Suppoter Vector Machine 


#### Methods

#### Results
Input n-gram tuple:(1,2)
logistic regression
Result of processed pipeline
Model Name: Logistic Regression Count Vectorizer
Accuracy: 0.56353591160221
Precision score: 0.56353591160221
F1 score: 0.7208480565371024
Recall score: 1.0
------------------------------------------------------------------------
Naive Bayes 
Result of Naive Bayes 
Model Name: Naive Bayes
Accuracy: 0.6195737963693765
Precision score: 0.6078066914498141
F1 score: 0.7307262569832402
Recall score: 0.9159663865546218
------------------------------------------------------------------------
SVM-high accuracy
Result of SVM
Model Name: SVM
Accuracy: 0.6195737963693765
Precision score: 0.6078066914498141
F1 score: 0.7307262569832402
Recall score: 0.9159663865546218

#### Discussion
