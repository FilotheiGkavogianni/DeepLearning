
from src.preprocessing.build_dataset import DataBuilder
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random

class Trainer:
    def __init__(self, model_type):
        self.history = None
        
        db = DataBuilder()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = db.get_data(model_type)


    def train(self, model):
        self.history = model.fit(self.X_train,
                                self.y_train,
                                epochs = 10,
                                batch_size = 512,
                                validation_data = (self.X_val, self.y_val))
                               


    def tune(self,model, model_type):
        if model_type == "mlp":
            save_dir = "mlp_dir"
        elif model_type == "cnn":
            save_dir = "cnn_dir"
        tuner = kt.Hyperband(model, # the hypermodel
                            objective='val_accuracy', # objective to optimize
                            max_epochs=10,
                            factor=3, # factor which you have seen above 
                            directory=save_dir, # directory to save logs 
                            project_name='khyperband')

        print(f"""
        The hyperparameter tuning search is iniating. The search space is the following {tuner.search_space_summary()}.
        """)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Perform hypertuning
        tuner.search(self.X_train, self.y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters()[0]
        if model_type == "mlp":
            print(f"""
            The hyperparameter search is complete. The optimal number of units in the first densely-connected
            layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}.
            """)
        elif model_type == "cnn":
            print(f"""
            The hyperparameter search is complete. The optimal number of units in the Conv2D
            layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}.
            """)
        h_model = tuner.hypermodel.build(best_hps)

        return h_model
    
    def predict(self,model):
        y_pred = model.predict(self.X_test)
        y_pred_argmax =  np.argmax(y_pred, axis=-1)
        return y_pred_argmax

    def predict_at_random(self, predictions):
        random_index = random.randint(0, len(predictions))
        random_prediction = predictions[random_index]
        random_image  = self.X_test[random_index].reshape( (28,28))
        random_correct = self.y_test[random_index]

        # plot raw pixel data
        fig, ax = plt.subplots(1,1)
        if random_correct == random_prediction:
            result = 'correctly'
        else:
            result = 'falsely'

        target_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        plt.title('The following image is '+ result +' predicted to be a : '+target_names[random_prediction], fontsize=8)
        plt.imshow(random_image, cmap=plt.get_cmap('gray'))
        
        # show the figure
        plt.show()


    def eval(self,model):
        eval = model.evaluate(self.X_test,self.y_test)
        print('Test loss is {}, Test accuracy is {}'.format(eval[0],eval[1]))

    def confusion_matrix(self, predictions):
        target_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        ConfusionMatrixDisplay.from_predictions(self.y_test,predictions, display_labels=target_names, xticks_rotation='vertical', cmap = 'RdPu')
        plt.show()

    def plot_metrics(self):
        print(self.history.history)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def class_report(self,predictions):
        target_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        print(classification_report(self.y_test, predictions, target_names=target_names))

