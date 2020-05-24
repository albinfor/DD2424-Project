from select_covid_patient_X_ray_images import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import numpy as np
from numpy import genfromtxt
import imageio
from tqdm import tqdm
from COVIDdata import COVIDdataset

def readDataFromFiles():
    covidData = COVIDdataset()
    y = genfromtxt('augmented_images/augmentation.csv', delimiter=',')
    for i in tqdm(range(len(y))):
        sample = {}
        X = imageio.imread('augmented_images/'+str(i)+'.png')
        sample['img'] = X.reshape(1, X.shape[0], X.shape[1])
        sample['lab'] = y[i]
        sample['idx'] = i
        covidData.add(sample)

    covidData.normalize()
    covidData.vectorize()
    covidData.generateMatrices()

    return covidData

if __name__ == "__main__":
    processpictures = False
    if processpictures or not os.path.isfile('Pickles/covidDatasetAugmented.p'):
        print(40 * "=")
        print("Creating object from files")
        print(40 * "=")
        covidset = readDataFromFiles()
        print()
        print(40 * "=")
        print("Completed creating object from files")
        pickle.dump(covidset, open("Pickles/covidDatasetAugmented.p", "wb"))
        print("Dataset Pickled")
    else:
        covidset = pickle.load(open("Pickles/covidDatasetAugmented.p", "rb"))

    covidset.y = to_categorical(covidset.y)
    X_train, X_test, y_train, y_test = train_test_split(covidset.X, covidset.y, test_size = 0.30)

    X_train = X_train.reshape(X_train.shape[0], covidset.minsize, covidset.minsize, 1)
    X_test = X_test.reshape(X_test.shape[0], covidset.minsize, covidset.minsize, 1)
    print("The distribution of the two classes are {}".format(np.mean(y_test,0)))
    # create model
    model = Sequential()
    # add model layers
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(covidset.minsize, covidset.minsize, 1)))
    #model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(covidset.minsize, covidset.minsize, 1)))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    #model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    #
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    # train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=35, batch_size = 153, callbacks=[checkpoint], verbose=True)

    print(model.evaluate(X_test, y_test))


    # Print confusion matrix
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("True negative: {} \nTrue positive: {} \nFalse negative: {} \nFalse positive: {}".format(tn,tp,fn,fp))

    pickle.dump(model, open("Pickles/latestmodelcnnaugmented.p", "wb"))

