from select_covid_patient_X_ray_images import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np

if __name__ == "__main__":
    processpictures = False
    if processpictures or not os.path.isfile('Pickles/covidDataset.p'):
        dataLoader = DataLoader(['PA'])
        print(40 * "=")
        print("Loading dataset from file")
        print(40 * "=")
        covidset = dataLoader.loadDataSet()
        print()
        print(40 * "=")
        print("Completed loading dataset from file")
        pickle.dump(covidset, open("Pickles/covidDataset.p", "wb"))
        print("Dataset Pickled")
    else:
        covidset = pickle.load(open("Pickles/covidDataset.p", "rb"))
    covidset.y = to_categorical(covidset.y)
    X_train, X_test, y_train, y_test = train_test_split(covidset.X, covidset.y, test_size = 0.30)

    X_train = X_train.reshape(X_train.shape[0], covidset.minsize, covidset.minsize, 1)
    X_test = X_test.reshape(X_test.shape[0], covidset.minsize, covidset.minsize, 1)

    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(covidset.minsize, covidset.minsize, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

    print(model.evaluate(X_test, y_test))


    # Print confusion matrix
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("True negative: {} \nTrue positive: {} \nFalse negative: {} \nFalse positive: {}".format(tn,tp,fn,fp))

    pickle.dump(model, open("Pickles/latestmodelaugment.p", "wb"))