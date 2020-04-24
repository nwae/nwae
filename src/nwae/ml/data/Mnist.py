# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.math.NumpyUtil import NumpyUtil
try:
    from keras.datasets import mnist
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing Keras modules: ' + str(ex_keras)
    )


class MnistData:

    @staticmethod
    def load_mnist_example_data():
        # Test data from MNIST, images of numbers 0-9
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Loading test data from MNIST..'
        )
        (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) =\
            mnist.load_data()

        n_samples = mnist_train_images.shape[0]
        n_pixels = NumpyUtil.get_point_pixel_count(mnist_train_images)
        # This will flatten the image to 1d, thus compressing to 2 dimensions, rows of pictures
        mnist_train_images_2d = mnist_train_images.reshape((n_samples, n_pixels))
        mnist_train_images_2d = mnist_train_images_2d.astype('float32') / 255
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Train images, total pixels = ' + str(n_pixels)
            + ', mnist_train_images shape = ' + str(mnist_train_images.shape)
            + ', mnist_train_images_2d shape = ' + str(mnist_train_images_2d.shape)
        )

        n_samples = mnist_test_images.shape[0]
        n_pixels = NumpyUtil.get_point_pixel_count(mnist_test_images)
        # This will flatten the image to 1d, thus compressing to 2 dimensions, rows of pictures
        mnist_test_images_2d = mnist_test_images.reshape((n_samples, n_pixels))
        mnist_test_images_2d = mnist_test_images_2d.astype('float32') / 255
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Test images, total pixels = ' + str(n_pixels)
            + ', mnist_test_images shape = ' + str(mnist_test_images.shape)
            + ', mnist_test_images_2d shape = ' + str(mnist_test_images_2d.shape)
        )

        return (mnist_train_images_2d, mnist_train_labels), (mnist_test_images_2d, mnist_test_labels)
