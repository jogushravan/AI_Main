import numpy
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.preprocessing import image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 2
lrate = 0.01
decay = lrate / epochs
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=2)

# Final evaluation of the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (test_acc * 100))

# Get predictions for all test samples
predictions = model.predict(X_test)

# Print accuracy, confusion matrix, and classification report
print("Accuracy:", accuracy_score(numpy.argmax(y_test, axis=1), numpy.argmax(predictions, axis=1)))
print("Confusion Matrix:\n", confusion_matrix(numpy.argmax(y_test, axis=1), numpy.argmax(predictions, axis=1)))
print("Classification Report:\n", classification_report(numpy.argmax(y_test, axis=1), numpy.argmax(predictions, axis=1)))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history[ 'loss' ])
plt.plot(history.history[ 'val_loss' ])
plt.title( 'model loss' )
plt.xlabel('Epoch')
plt.ylabel( 'loss' )
plt.legend([ 'train' , 'test' ], loc='upper left' )
plt.show()


# Load a new image for prediction
new_img_path = X_test[10]
new_img = image.load_img(new_img_path, target_size=(32, 32))
new_img_array = image.img_to_array(new_img)
new_img_array = numpy.expand_dims(new_img_array, axis=0)  # Add batch dimension
new_img_array = new_img_array / 255.0  # Normalize the pixel values

# Make prediction
prediction = model.predict(new_img_array)
predicted_class_index = numpy.argmax(prediction)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class_label = class_labels[predicted_class_index]

print("Predicted Class:", predicted_class_label)
print("Prediction Probabilities:", prediction)