import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda")

# df = pd.read_csv("spamdata_v2.csv")
# df.head()



def Open_DataSet():
    df = pd.read_csv("emails.csv")
    print(df['Prediction'].value_counts(normalize = True))
    X = df.iloc[:,1:3001].values
    Y = df.iloc[:,-1].values
    return X,Y

def Bert(X,Y,valueTest_size):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = valueTest_size)

X,Y = Open_DataSet()
# print(X)

bert = AutoModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
