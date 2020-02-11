"""
Dataset1: Liar dataset
LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.
=====================================================================
Description of the TSV format:

Column 1: the ID of the statement ([ID].json).
Column 2: the label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).

Note that we do not provide the full-text verdict report in this current version of the dataset,
but you can use the following command to access the full verdict report and links to the source documents:
wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json

======================================================================
The original sources retain the copyright of the data.

Note that there are absolutely no guarantees with this data,
and we provide this dataset "as is",
but you are welcome to report the issues of the preliminary version
of this data.

You are allowed to use this dataset for research purposes only.

For more question about the dataset, please contact:
William Wang, william@cs.ucsb.edu

v1.0 04/23/2017



this dataset is the same used in named search:

"Liar, Liar Pants on Fire":
The New Benchmark Dataset for Fake News Detection

LINK: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

2019-03-02







"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Reading file helper

def read_tsv(path):
    """
    Use panda to read in .tsv files
    :param path: file path points to .tsv file
    :return:A comma-separated values (csv) file
    is returned as two-dimensional data structure with labeled axes.
    """
    #\t in a string literal is an escape sequence
    # for tab character, horizontal whitespace,
    # ASCII codepoint 9.
    dataframe = pd.read_csv(path,sep='\t',header=None)
    return dataframe

def read_csv_file(path):
    """
    read in a .csv file
    :param path: file path points to .csv file
    :return: panda dataframe type
    """
    dataframe = pd.read_csv(path)
    return dataframe

# Pre-processing
def pre_process_data(dataframe):
    """
    1.Obtain a pure 2 coloumn form of the original dataset
    2.Convert string into lower case
    :param dataframe: panda dataframe seperated by space
    :return: a data frame with desired coloumn selected.
    """
    wanted_col = [1,2]
    """
    An integer, e.g. 5.
    A list or array of integers, e.g. [4, 3, 0].
    A slice object with ints, e.g. 1:7.
    A boolean array.
    A callable function
    """
    #1.Obtain a pure 2 coloumn form of the original dataset
    dt_selected = dataframe.iloc[:,wanted_col]
    dt_selected.columns = ['label','news']
    #2.Convert string into lower case
    dt_selected['news'] = dt_selected['news'].str.lower()
    #3.Remove invalid row
    """
    isnull() detect missing values in the dataframe
    sum() Return the sum of the values for the requested axis.
    """
    has_null = dt_selected.isnull().sum().sum()
    if has_null > 0:
        dt_selected.dropna()
    return dt_selected

# Generate processed dataframe

def csv_gen(dataframe,path,name):
    """
     After pre-process and selection,generate a pre-processed file
    :param dataframe: a pre-processed dataframe in panda format
    :param path: a file path where you want to save
    :param name: a file name for saving the file
    :return: None
    """
    path_name = path + name
    'Write object to a comma-separated values (csv) file.'
    dataframe.to_csv(path_name,index=False)

# lambda function pipeline
def func_compose(a,b,c,var1,var2):
    """
    Lambda function pipeline
    :param a: A function
    :param b: A function
    :param c: A function
    :param d: A function
    :return: Final result spit out from pipeline.
    """
    return lambda x:a(b(c(x)),var1,var2)
# Display the data frame
def display_data(path):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pre_process_data(read_tsv(path)))

# Display dataframe in word cloud
def df_word_cloud(name,save_to,path=None,df_read=None):
    if(path is None):
        text = df_read.news.values
    else:
        df = read_csv_file(path)
        text = df.news.values
    wordcloud = WordCloud(
        width=3000,
        height=2000,
        background_color='black',
        stopwords=STOPWORDS).generate(str(text))
    path_name = save_to + name
    wordcloud.to_file(path_name)
    # fig = plt.figure(
    #     figsize=(40, 30),
    #     facecolor='k',
    #     edgecolor='k')
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # plt.show()

# File paths:
# Data file paths:
train = "./datasets/Original_muti/train.tsv"
valid = "./datasets/Original_muti/valid.tsv"
test = "./datasets/Original_muti/test.tsv"
# Where to save:
save_to = "./datasets/v1_dataset/"
# -----------------------------------------------
# All convert to data frame format
# Pre-process
# Saving the dataframe into csv
# Lower case format into csv
# Check for invalid row
#
# func_compose(csv_gen,pre_process_data,read_tsv,save_to,"train.csv")(train)
# func_compose(csv_gen,pre_process_data,read_tsv,save_to,"valid.csv")(valid)
# func_compose(csv_gen,pre_process_data,read_tsv,save_to,"test.csv")(test)

#display as wordcloud
# name saveto path
# df_word_cloud("wc_train.png","./graph/","./datasets/v1_dataset/train.csv")
# df_word_cloud("wc_valid.png","./graph/","./datasets/v1_dataset/valid.csv")
# df_word_cloud("wc_test.png","./graph/","./datasets/v1_dataset/test.csv")

train_v1 = read_csv_file("./datasets/v1_dataset/train.csv")
valid_v1 = read_csv_file("./datasets/v1_dataset/valid.csv")
test_v1 = read_csv_file("./datasets/v1_dataset/test.csv")
tio = pd.concat([train_v1,valid_v1,test_v1],ignore_index=True)

labels = ['true','mostly-true','half-true','false','barely-true','pants-fire']
for x in labels:
    df_word_cloud("wc" + x + ".png", "./graph/", df_read=tio[tio['label'] == x])



# 	true
# mostly-true	true
# half-true	true
# false	false
# barely-true	false
# pants-fire


