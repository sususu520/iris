from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPool1D, Dropout

import matplotlib.pyplot as plt
import pandas as pd

# load iris data
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

#  number of label classes
num_classes = len(pd.Series(iris.target).unique())
# print(num_classes)

#  label's label encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# label's one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# reshape 2D to 3D ->  x_train.reshape(num_of_examples,num_of_features,num_of_signals)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# build CNN model
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(4, 1), activation='relu'))  # convolution
model.add(MaxPool1D(pool_size=2))  # pooling
model.add(Flatten())  # flatten
model.add(Dense(128, activation='relu'))  # fc
model.add(Dropout(0.3))  # dropout
model.add(Dense(num_classes, activation='softmax'))

# model compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

# model training
batch_size = 1
epochs = 1000
model = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_test, y_test))

# draw loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()