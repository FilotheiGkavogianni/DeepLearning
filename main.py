
from tokenize import String
from sklearn import model_selection
from src.training.trainer import Trainer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import os
import sys
import numpy as np
from src.models.mlp import MLP
from src.models.cnn import CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras_tuner as kt
import os


def main(model_type, tune):

    pixels = 28*28
    num_categories = 10
    if model_type == 'mlp':
        model = MLP(pixels,num_categories)

        if tune == False:
            model_object = model.build_model()

            trainer = Trainer(model_type)

            trainer.train(model_object)

        elif tune == True:
            model_builder = model.model_builder

            trainer = Trainer(model_type)

            model_object = trainer.tune(model_builder)
            
            trainer.train(model_object)


        trainer.plot_metrics()

        trainer.eval(model_object)

        predictions = trainer.predict(model_object)

        trainer.confusion_matrix(predictions)

        trainer.class_report(predictions)

    elif model_type == 'cnn':
        model = CNN(pixels,num_categories)

        if (tune == False):

            model_object = model.build_model()

            trainer = Trainer(model_type)

            trainer.train(model_object)
        trainer.plot_metrics()

        trainer.eval(model_object)

        predictions = trainer.predict(model_object)

        trainer.confusion_matrix(predictions)

        trainer.class_report(predictions)

        pass
    


if __name__ == "__main__":
    model_type = str(sys.argv[1])
    tune = eval(sys.argv[2])
    print(model_type)
    print(tune)
    main(model_type,tune)

#
#https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/