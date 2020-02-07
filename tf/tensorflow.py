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
