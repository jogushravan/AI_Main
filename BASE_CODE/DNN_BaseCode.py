from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier
import numpy
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold , cross_val_score
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",", skiprows=1)

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_model():
    model = Sequential()
    model.add(Input(shape=(8,)))  # Define input shape explicitly
    model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

estimators = []
estimators.append(( 'standardize' , StandardScaler()))
estimators.append(( 'mlp' , DecisionTreeClassifier(build_fn=create_model, nb_epoch=300, batch_size=16, verbose=0)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# # checkpoint    
# filepath = "weights-improvement.keras"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# # Fit the model
# history =model.fit(X, Y, validation_split=0.33, epochs=20, batch_size=10,callbacks=callbacks_list, verbose=0)
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title( 'data=model accuracy' )
# plt.ylabel('accuracy'  )
# plt.xlabel('epoch'  )
# plt.legend([ 'train' , 'test' ], loc= 'upper left' )
# plt.show()
# # summarize history for loss
# plt.plot(history.history[ 'loss' ])
# plt.plot(history.history[ 'val_loss' ])
# plt.title( 'model loss' )
# plt.ylabel( 'loss' )
# plt.xlabel( 'epoch' )
# plt.legend([ 'train' , 'test' ], loc='upper left' )
# plt.show()

