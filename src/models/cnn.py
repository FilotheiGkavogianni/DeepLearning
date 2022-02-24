from matplotlib.cbook import flatten
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import keras_tuner as kt

class CNN():

    def __init__(self, in_shape, num_categories):
        self.in_shape = in_shape
        self.num_categories = num_categories
        pass

    def build_model(self):
        # create model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(self.num_categories, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=['accuracy'],
        )

        return self.model
