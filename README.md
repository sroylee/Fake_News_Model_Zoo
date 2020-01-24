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
1.Liar Dataset

2.Preprocessed Liar Dataset

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

---

### Traditional Methods
Logistic Regression
Naive Bayes 
Suppoter Vector Machine 
Decision Tree
Random Forest

#### Methods

#### Results

#### Discussion
