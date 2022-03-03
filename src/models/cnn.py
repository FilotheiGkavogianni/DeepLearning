from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import keras_tuner as kt

class CNN():

    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories

    def baseline_model(self):
        # create model
        self.model = Sequential()
        self.model.add(Conv2D(12, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        #self.model.add(MaxPooling2D((2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(self.num_categories, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=['accuracy'],
        )

        return self.model


    def model_builder(self,hp):
        
        model = keras.Sequential()

        # Tune the number of units in the first Conv2D layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=2, max_value=32, step=4)

        model.add(Conv2D(hp_units, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

        model.add(Flatten())
        model.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.num_categories, activation='softmax'))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            metrics=['accuracy'],
        )

        return model

    