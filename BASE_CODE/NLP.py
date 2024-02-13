import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.preprocessing import sequence
from keras.activations import relu, sigmoid
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Conv1D,Conv2D, MaxPool2D,MaxPool1D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32)) #dropout=0.2)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(LSTM(100))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict(X_test)
# Convert probabilities to binary predictions
binary_predictions = (predictions > 0.5).astype(int)
#___________________________________________________________________
print("Accuracy:", accuracy_score(y_test, binary_predictions))
print(confusion_matrix(y_test, binary_predictions))
print(classification_report(y_test, binary_predictions))