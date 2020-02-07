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
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normalizer(train_data, test_data)

#Now on Machine Learning , we will be using Keras API 
#we wil; build our model in 2layers .
#user Rectified linear Unit and Logistic regression on our model to 2 probability groups and labels 
length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))


#compiling, evaluating and fitting data into the model 
#pick an optimizer and loss to show the training rate
#I used the SGD Algorithim  fro optimization
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#now fit in the model with Training data, training label ad epoch . as it is supervised Machine Learning;
model.fit(train_data, train_labels, epochs=400)

#now that  the model is trained, lets test with oue test data 
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value})'
>>> Our test accuracy was 0.980132 which is not tha bod, almost perfect accuracy

#now predict a given pokemon  with model.predict
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return(prediction)
        
#Let's look at one of the more well-known legendary Pokémon: Mewtwo.
#  He's number 150 in the list of Pokémon, so we'll look at index 149:

predictor(test_data, test_labels, 149)
>>> This was correctly predicted to be a "1"!
