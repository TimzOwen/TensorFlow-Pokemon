#install tensorflow
pip install tensorflow

#import required libraries 

import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing 

# Obtain data from kaggle of pretrained pekon "pikachu" fro prediction of our model
#read the cvs file using the pandas
df = pd.read_cvs('C:\Users\Owe\Desktop\tf\pokemon')
#check for boolean  0/1 for the legendary player 
df['isLegendary'] = df['isLegendary'].astype(int)

#create dummy variables to hold the attacks techniques such as water n grass etc
#pd.get_dummies creates a new dataFrame from the categories
#the dummy variables makes sure we don't alloct intergers not relevant to the game properties
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

#Upnext Splitting and normalizing data
#we will split our data into training and testing data, Let's use pekomen generation of
#blue, red and yellow fro training
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)

df_train, df_test = train_test_splitter(df, 'Generation')

# separate labels from the data sets as category 1 and the rest
def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

#extract the data from the datasets and out them into arrays which tensorflow can understand
#usinf .values
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')

#upnext is normalizing the data so that each pikachu is on the same scale
