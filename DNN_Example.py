from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras.layers import BatchNormalization
import keras
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
df = df.fillna(0)

seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

num_epochs = 30

def build_model_DNN(featuresize, neurons, newneurons2, drop, drop2, lr):
  initializer = tf.keras.initializers.GlorotUniform(seed=10)
  initializer2 = tf.keras.initializers.GlorotUniform(seed=20)
  initializer3 = tf.keras.initializers.GlorotUniform(seed=30)

  model = models.Sequential()
  model.add(layers.Dense(neurons, activation='relu', kernel_initializer=initializer, input_shape=(featuresize,)))
  model.add(BatchNormalization())
  model.add(layers.Dropout(drop))
  model.add(layers.Dense(newneurons2, activation='relu', kernel_initializer=initializer2))
  model.add(BatchNormalization())
  model.add(layers.Dropout(drop2))
  model.add(layers.Dense(3, kernel_initializer=initializer3, activation='softmax'))
  opt = keras.optimizers.Adam(learning_rate=lr)
  model.compile(optimizer='Adam', loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=['accuracy'])
  return model

fs = SelectKBest(score_func=f_classif, k=30)
X = fs.fit_transform(df, y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=22, stratify=y)

label_encoder = LabelEncoder()

integer_encoded_train_y= label_encoder.fit_transform(train_y)
integer_encoded_test_y= label_encoder.fit_transform(test_y)

encoded_train_y = to_categorical(integer_encoded_train_y)
encoded_test_y = to_categorical(integer_encoded_test_y)


#DNN Generating and Training
#model = build_model_DNN(featuresize, newneurons, newneurons2, newdrop, newdrop2, newlearningrate)
model = build_model_DNN(30, 128, 32, 0.1, 0.3, 0.01)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=0, restore_best_weights=True)

model.fit(train_X, encoded_train_y, epochs=num_epochs, batch_size=8, verbose=0, callbacks=[early_stop])

model.save('DNN.h5')

#DNN Predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis=1)

matrix = confusion_matrix(integer_encoded_test_y, y_pred)

accuracy = accuracy_score(integer_encoded_test_y, y_pred)
precision_score = precision_score(integer_encoded_test_y, y_pred, average='weighted')
recall_score = recall_score(integer_encoded_test_y, y_pred, average='macro')
f1_score = f1_score(integer_encoded_test_y, y_pred, average='weighted')

print("Accuracy : " + str(accuracy))
print("Precision : " + str(precision_score))
print("Recall : " + str(recall_score))
print("f1_score : " + str(f1_score))

plt.clf()
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('P-0s', 'P-1s','P-2s'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('A-0s', 'A-1s','A-2s'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='red')

plt.savefig('CM_Predicted_class3.png')
plt.show()
