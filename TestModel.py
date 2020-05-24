from keras.models import load_model
from keras.utils import to_categorical
import pickle
import os
import logging

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    covidset = pickle.load(open("Pickles/covidDataset.p", "rb"))
    model = load_model('Pickles/best_model_82.hdf5')

    covidset.X = covidset.X.reshape(covidset.X.shape[0], covidset.minsize, covidset.minsize, 1)
    covidset.y = to_categorical(covidset.y)

    y_pred = model.predict(covidset.X)
    scores = model.evaluate(covidset.X, covidset.y, verbose=True)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.summary()

