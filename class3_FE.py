from google.colab import drive

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

csv = '/content/gdrive/My Drive/Colab Notebooks/Vehicle Type Classification/class3_FE.csv'
csv_class = '/content/gdrive/My Drive/Colab Notebooks/Vehicle Type Classification/class3.csv'
drive.mount('/content/gdrive')

df = pd.read_csv(csv, header=None)

df_class = pd.read_csv(csv_class, header=None)
df_class = df_class[0].str.split(";",expand=True,) # splitting columns
train_y = df_class[621]

new_df = pd.concat([df, train_y], axis=1, join="inner") # we added labels to the FE dataset
new_df = new_df.sample(frac=1, random_state = 20) # shuffle samples
df = new_df 
df = df.drop(columns=[44, 45, 46, 47, 48, 49])
train_y = df[621]
del df[621]

train_X = df.values 
min_max_scaler = preprocessing.MinMaxScaler()
train_X = min_max_scaler.fit_transform(train_X)
train_X = pd.DataFrame(train_X)
train_X

print("-Print Shapes of Dataframe-")
print("X:", train_X.shape, " - Y:", train_y.shape)

train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size = 0.3, random_state=22, stratify=train_y)

print("train_X: " + str(train_X.shape))
print("train_y: " + str(train_y.shape))
print("test_X: " + str(test_X.shape))
print("test_y: " + str(test_y.shape))

label_encoder = LabelEncoder()

integer_encoded_train_y= label_encoder.fit_transform(train_y)
integer_encoded_test_y= label_encoder.fit_transform(test_y)

encoded_train_y = to_categorical(integer_encoded_train_y)
encoded_test_y = to_categorical(integer_encoded_test_y)

train_X = train_X.fillna(0)
test_X = test_X.fillna(0)
