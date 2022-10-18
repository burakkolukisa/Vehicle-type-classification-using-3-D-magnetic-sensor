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

model = build_model_DNN(featuresize, newneurons, newneurons2, newdrop, newdrop2, newlearningrate)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=0, restore_best_weights=True)

model.fit(train_X, encoded_train_y, epochs=num_epochs, batch_size=8, verbose=0, callbacks=[early_stop])

y_pred = model.predict(test_X)
