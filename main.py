
from src.training.trainer import Trainer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import os
import numpy as np
from src.models.mlp import MLP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras_tuner as kt
import os


def main(model_type, tune, task):

    pixels = 28*28
    num_categories = 10
    if model_type == 'mlp':
        model = MLP(pixels,num_categories)
    elif model_type == 'cnn':
        #model = CNN(pixels,num_categories)
        pass
    
    if tune == False:
        model_object = model.build_model()

        trainer = Trainer()

        trainer.train(model_object)


    elif tune == True:
        model_builder = model.model_builder

        trainer = Trainer()

        model_object = trainer.tune(model_builder)
        
        trainer.train(model_object)


    trainer.plot_metrics()

    trainer.eval(model_object)

    predictions = trainer.predict(model_object)

    trainer.confusion_matrix(predictions)

    trainer.class_report(predictions)
    
    # if model == 'mlp' and tune == False :
        

        
        
        

    #     eval = mlp.evaluate(X_train,y_train)
    #     print('Val loss is {}, Val accuracy is {}'.format(eval[0],eval[1]))

    #     eval = mlp.evaluate(X_test,y_test)
    #     print('Test loss is {}, Test accuracy is {}'.format(eval[0],eval[1]))
    #     predictions = mlp.predict(X_test)
    #     for i in range(20):
    #         print(np.argmax(predictions[i]),y_test[i])

    #     y_pred =  np.argmax(predictions, axis=-1)

        
    #     # db.plot_image(0)

    # elif model == 'mlp' and tune == True :
    #     print('here')
    #     mlp = MLP(pixels,num_categories).model_builder

    #     tuner = kt.Hyperband(mlp, # the hypermodel
    #                         objective='val_accuracy', # objective to optimize
    #                         max_epochs=10,
    #                         factor=3, # factor which you have seen above 
    #                         directory='dir', # directory to save logs 
    #                         project_name='khyperband')
        
    #     print(tuner.search_space_summary() )

    #     stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #     # Perform hypertuning
    #     tuner.search(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

    #     best_hps = tuner.get_best_hyperparameters()[0]
    #     print(f"""
    #     The hyperparameter search is complete. The optimal number of units in the first densely-connected
    #     layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    #     is {best_hps.get('learning_rate')}.
    #     """)

    #     h_model = tuner.hypermodel.build(best_hps)
    #     print(h_model.summary())



if __name__ == "__main__":
    main('mlp',True, 'train')

#
#https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/