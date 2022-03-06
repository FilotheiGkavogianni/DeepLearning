from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt

class MLP():
    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories
        #self.model = self.build_model(in_shape,num_categories)
        
    def baseline_model(self):
        # create model
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.in_shape, activation='relu'))
        self.model.add(Dropout(0.10))
        self.model.add(Dense(self.num_categories, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            metrics=['accuracy'],
        )

        return self.model


    def model_builder(self,hp):
        
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, input_dim=self.in_shape, activation='relu'))
        # Add next layers
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(self.num_categories, activation='softmax'))
        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer= keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        return model

    