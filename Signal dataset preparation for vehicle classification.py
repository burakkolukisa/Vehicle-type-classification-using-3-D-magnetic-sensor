from google.colab import drive

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

csv = '/content/gdrive/My Drive/Colab Notebooks/Vehicle Classification/class3.csv'
drive.mount('/content/gdrive')

df = pd.read_csv(csv, header=None)
df = df.sample(frac=1, random_state = 20) # shuffle samples
df = df[0].str.split(";",expand=True,) # splitting columns
train_y = df[621]
print("-Number of samples for each class-")
print(df.groupby(621).size())
del df[621]

train_X = df.values 
min_max_scaler = preprocessing.MinMaxScaler()
train_X = min_max_scaler.fit_transform(train_X)
df = pd.DataFrame(train_X)

print("-Print Shapes of Dataframe-")
print("X:", train_X.shape, " - Y:", train_y.shape)


train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size = 0.30, random_state=22, stratify=y)

print("train_X: ", train_X.shape)
print("train_y: ", train_y.shape)
print("test_X: ", test_X.shape)
print("test_y: ", test_y.shape)

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

integer_encoded_train_y= label_encoder.fit_transform(train_y)
integer_encoded_test_y= label_encoder.fit_transform(test_y)

encoded_train_y = to_categorical(integer_encoded_train_y)
encoded_test_y = to_categorical(integer_encoded_test_y)
