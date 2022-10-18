from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_addons as tfa
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras.layers import BatchNormalization
import keras
import warnings

warnings.filterwarnings('ignore')
df = df.fillna(0)

seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

num_epochs = 30
num_val_samples = len(train_X) // 5 # 5-Fold

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

def build_optimization(featuresize, newneurons, newneurons2, newdrop, newdrop2, newbatchsize, newlearningrate):
  total = 0 
  for i in range(k):
    val_X = train_X[i * num_val_samples: (i + 1) * num_val_samples]
    val_y = train_y[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_X = np.concatenate([train_X[:i * num_val_samples],train_X[(i + 1) * num_val_samples:]],axis=0)
    partial_train_y = np.concatenate([train_y[:i * num_val_samples],train_y[(i + 1) * num_val_samples:]],axis=0)

    label_encoder = LabelEncoder()

    integer_encoded_val_y= label_encoder.fit_transform(val_y)
    integer_encoded_partial_train_y= label_encoder.fit_transform(partial_train_y)

    encoded_val_y = to_categorical(integer_encoded_val_y)
    encoded_partial_train_y = to_categorical(integer_encoded_partial_train_y)

    model = build_model_DNN(featuresize, newneurons, newneurons2, newdrop, newdrop2, newlearningrate)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=0, restore_best_weights=True)

    model.fit(partial_train_X, encoded_partial_train_y, epochs=num_epochs, batch_size=newbatchsize, verbose=0, callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(val_X, encoded_val_y, verbose=0)

    total = total + test_acc

  return total/k


for i in range(1, 44):
  global_max_acc = 0
  from sklearn.preprocessing import LabelEncoder
  from sklearn.preprocessing import OneHotEncoder
  from tensorflow.keras.utils import to_categorical
  from sklearn.model_selection import train_test_split

  fs = SelectKBest(score_func=f_classif, k=i)
  X = fs.fit_transform(df, y)

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.30, random_state=22, stratify=y)

  label_encoder = LabelEncoder()
  integer_encoded_train_y= label_encoder.fit_transform(train_y)
  encoded_train_y = to_categorical(integer_encoded_train_y)

  neurons = [32, 64, 128]
  neurons2 = [32, 64, 128]
  drop = [0.1, 0.3, 0.5]
  drop2 = [0.1, 0.3, 0.5]
  batchsize = [2, 4, 6, 8]
  lr = [10e-2, 10e-3, 10e-4]

  for n in neurons:
    for n2 in neurons2:
      for d in drop:
        for d2 in drop2:
          for b in batchsize:
            for l in lr:
              total = 0
              x = build_optimization(i, n, n2, d, d2, b, l)
              if x >= global_max_acc:                       
                global_max_acc = x 
                print(str(i) + " - " + str(n) + "-" + str(n2) + "-" + str(d) + "-" +  str(d2) + "-" + str(b)+ "-" + str(l) + "- : Accuracy : " + str(x))
