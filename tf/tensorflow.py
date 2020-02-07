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