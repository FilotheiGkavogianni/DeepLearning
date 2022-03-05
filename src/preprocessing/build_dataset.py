

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

class DataBuilder():

    def get_data(self,model_type):
        
        # load dataset
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        
        # split to get validation set
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.17)

        if (model_type == "mlp"):
            train_images_reshape = train_images.reshape(train_images.shape[0], 28*28)
            val_images_reshape = val_images.reshape(val_images.shape[0], 28*28)
            test_images_reshape = test_images.reshape(test_images.shape[0], 28*28)
        elif (model_type == "cnn"):
            train_images_reshape = train_images.reshape((train_images.shape[0], 28, 28, 1))
            val_images_reshape = val_images.reshape((val_images.shape[0], 28, 28, 1))
            test_images_reshape = test_images.reshape((test_images.shape[0], 28, 28, 1))
        
        
        self.train_images = train_images_reshape.astype('float32') / 255.0
        self.train_labels = train_labels

        self.val_images = val_images_reshape.astype('float32') / 255.0
        self.val_labels = val_labels

        self.test_images = test_images_reshape.astype('float32') / 255.0
        self.test_labels = test_labels

        

        print('Shape of X_train, y_train is {} {}'.format(self.train_images.shape, self.train_labels.shape))
        print('Shape of X_val, y_val is {} {}'.format(self.val_images.shape, self.val_labels.shape))
        print('Shape of X_test y_test is {} {}'.format(self.test_images.shape, self.test_labels.shape))

        return self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels 


    def plot_image(self,i):
        # plot raw pixel data
        pyplot.imshow(self.train_images[i], cmap=pyplot.get_cmap('gray'))
        # show the figure
        return pyplot.show()

