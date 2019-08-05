# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import mozg.common.util.ObjectPersistence as objper


class Image_Bw:

    def __init__(self):
        return

    def load_network(
            self
    ):
        b = ObjectPersistence.deserialize_object_from_file(
            obj_file_path=obj_file_path,
            lock_file_path=lock_file_path,
            verbose=3
        )

    def train(
            self,
            network_file_path
    ):
        (train_images, train_labels),(test_images, test_labels) = mnist.load_data()

        print('Data type train images "' + str(type(train_images)) + '" shape ' + str(train_images.shape)
              + ', train labels "' + str(type(train_labels)) + '", shape ' + str(train_labels.shape))

        network = models.Sequential()
        network.add(
            layers.Dense(
                units=512,
                activation = 'relu',
                input_shape = (28*28,)
            )
        )
        network.add(
            layers.Dense(
                units=10,
                activation='softmax'
            )
        )

        network.compile(
            optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )

        train_images = train_images.reshape((60000, 28*28))
        train_images = train_images.astype('float32') / 255

        test_images = test_images.reshape((10000, 28*28))
        test_images = test_images.astype('float32') / 255

        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        network.fit(
            train_images, train_labels, epochs=5, batch_size=128
        )

        objper.ObjectPersistence.serialize_object_to_file(
            obj = network,
            obj_file_path  = network_file_path,
            lock_file_path = None
        )

        test_loss, test_acc = network.evaluate(test_images, test_labels)
        print('Test accuracy: ', test_acc)

        prd = network.predict_classes(x=test_images[0:10])
        print(prd)

        return


if __name__ == '__main__':
    obj = Image_Bw()
    obj.train(network_file_path='/Users/mark.tan/git/mozg/app.data/general/example.kr.network')