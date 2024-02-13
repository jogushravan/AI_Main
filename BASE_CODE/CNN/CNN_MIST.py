# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten, Dropout, Conv2D, MaxPool2D
from keras.utils import to_categorical
import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Flattening the images from the 28x28 pixels to 2D 787 pixels
# normalizing the data to help with the training
# building the input vector from the 28x28 pixels
X_train = X_train.astype('float32')/ 255.0
X_test = X_test.astype('float32')/ 255.0
# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 10 epochs

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath='data/model_callback.weights.h5',
    save_weights_only=True,
    monitor='val_loss',  # 'val_accuracy'
    mode='min',  # 'max'
    save_best_only=True,
    verbose=1)

earlystop_cb = keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    verbose=1)

callbacks = [earlystop_cb, checkpoint_cb]

model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_data=(X_test, Y_test),callbacks=callbacks,verbose=1)

