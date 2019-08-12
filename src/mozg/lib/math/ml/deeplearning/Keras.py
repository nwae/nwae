# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import mozg.lib.math.ml.TrainingDataModel as tdm
import mozg.utils.Log as log
from inspect import currentframe, getframeinfo
import mozg.lib.math.ml.ModelInterface as modelIf
import mozg.utils.ObjectPersistence as objper
import mozg.lib.math.NumpyUtil as npUtil
import matplotlib.pyplot as plt


class Keras(modelIf.ModelInterface):

    MODEL_NAME = 'keras_nn'

    def __init__(
            self,
            # Unique identifier to identify this set of trained data+other files after training
            identifier_string,
            # Directory to keep all our model files
            dir_path_model,
            # Training data in TrainingDataModel class type
            training_data = None,
            do_profiling = False
    ):
        super(Keras,self).__init__(
            model_name        = Keras.MODEL_NAME,
            identifier_string = identifier_string,
            dir_path_model    = dir_path_model,
            training_data     = training_data
        )
        self.identifier_string = identifier_string
        self.dir_path_model = dir_path_model
        self.training_data = training_data
        if self.training_data is not None:
            self.__check_training_data()

        self.filepath_model = modelIf.ModelInterface.get_model_file_prefix(
            dir_path_model    = self.dir_path_model,
            model_name        = self.model_name,
            identifier_string = self.identifier_string
        )
        self.network = None

        self.do_profiling = do_profiling

        # Testing only
        (self.mnist_train_images, self.mnist_train_labels), (self.mnist_test_images, self.mnist_test_labels) =\
            (None, None), (None, None)
        self.mnist_train_images_2d = None
        self.mnist_test_images_2d = None
        return

    def set_training_data(
            self,
            td
    ):
        self.training_data = td
        self.__check_training_data()

    def __check_training_data(self):
        if type(self.training_data) is not tdm.TrainingDataModel:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Training data must be of type "' + str(tdm.TrainingDataModel.__class__)
                + '", got type "' + str(type(self.training_data))
                + '" instead from object ' + str(self.training_data) + '.'
            )

    def get_model_features(
            self
    ):
        return None
        #raise Exception('How to find the number of columns in the network?')

    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x
    ):
        p = self.network.predict_classes(x=x)
        return Keras.predict_class_retclass(
            predicted_classes = p
        )

    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x
    ):
        p = self.network.predict_classes(x=x)
        return Keras.predict_class_retclass(
            predicted_classes = p
        )

    def train(
            self,
            persist_model_to_storage = True,
            persist_training_data_to_storage = False,
            model_params = None,
    ):
        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Training for data, x shape '  + str(self.training_data.get_x().shape)
            + ', train labels with shape ' + str(self.training_data.get_y().shape)
        )

        x = self.training_data.get_x().copy()
        y = self.training_data.get_y().copy()
        train_labels_categorical = to_categorical(y)

        n_labels = len(list(set(y.tolist())))
        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Total unique labels = ' + str(n_labels) + '.'
        )

        network = models.Sequential()

        i_layer = 1
        for ly in model_params:
            if i_layer == 1:
                network.add(
                    layers.Dense(
                        units       = ly['units'],
                        activation  = ly['activation'],
                        input_shape = ly['input_shape']
                    )
                )
            else:
                network.add(
                    layers.Dense(
                        units      = ly['units'],
                        activation = ly['activation']
                    )
            )
            i_layer += 1

        network.compile(
            optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )

        # Log model summary
        network.summary(print_fn=log.Log.info)

        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + 'Categorical Train label shape "' + str(train_labels_categorical.shape)
            + '":\n\r' + str(train_labels_categorical)
        )

        network.fit(
            x, train_labels_categorical, epochs=5, batch_size=128
        )

        self.network = network

        if persist_model_to_storage:
            self.persist_model_to_storage(network=network)
        if persist_training_data_to_storage:
            self.persist_training_data_to_storage(td=self.training_data)
        return

    def load_model_parameters(
            self
    ):
        self.network = objper.ObjectPersistence.deserialize_object_from_file(
            obj_file_path = self.filepath_model
        )

    def persist_model_to_storage(
            self,
            network = None
    ):
        objper.ObjectPersistence.serialize_object_to_file(
            obj = network,
            obj_file_path  = self.filepath_model,
            lock_file_path = None
        )
        log.Log.info(
            str(self.__class__) + str(getframeinfo(currentframe()).lineno)
            + ': Saved network to file "' + self.filepath_model + '".'
        )
        return

    def load_mnist_example_data(self):
        # Test data from MNIST
        log.Log.info('Loading test data from MNIST..')
        (self.mnist_train_images, self.mnist_train_labels), (self.mnist_test_images, self.mnist_test_labels) =\
            mnist.load_data()

        n_samples = self.mnist_train_images.shape[0]
        n_pixels = npUtil.NumpyUtil.get_point_pixel_count(self.mnist_train_images)
        log.Log.debugdebug('Train images, total pixels = ' + str(n_pixels))
        self.mnist_train_images_2d = self.mnist_train_images.reshape((n_samples, n_pixels))
        self.mnist_train_images_2d = self.mnist_train_images_2d.astype('float32') / 255

        n_samples = self.mnist_test_images.shape[0]
        n_pixels = npUtil.NumpyUtil.get_point_pixel_count(self.mnist_test_images)
        log.Log.debugdebug('Test images, total pixels = ' + str(n_pixels))
        self.mnist_test_images_2d = self.mnist_test_images.reshape((n_samples, n_pixels))
        self.mnist_test_images_2d = self.mnist_test_images_2d.astype('float32') / 255

        return


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO

    kr = Keras(
        identifier_string = 'keras_image_bw_example',
        dir_path_model    = '/Users/mark.tan/git/mozg/app.data/models'
    )
    kr.load_mnist_example_data()

    do_train = True

    if do_train:
        td = tdm.TrainingDataModel(
            x = kr.mnist_train_images_2d,
            y = kr.mnist_train_labels,
            is_map_points_to_hypersphere = False
        )
        kr.set_training_data(td = td)

        #
        # Layers Design
        #
        nn_layers = [
            {
                'units': 512,
                'activation': 'relu',
                'input_shape': (td.get_x().shape[1],)
            },
            {
                'units': to_categorical(td.get_y()).shape[1],
                'activation': 'softmax'
            }
        ]

        print('Training started...')
        kr.train(
            model_params = nn_layers
        )
        print('Training done.')

    print('Loading model parameters...')
    kr.load_model_parameters()
    test_labels_cat = to_categorical(kr.mnist_test_labels)

    test_loss, test_acc = kr.network.evaluate(kr.mnist_test_images_2d, test_labels_cat)
    print('Test accuracy: ', test_acc)

    prd = kr.predict_classes(x=kr.mnist_train_images_2d[0:10])
    print(prd.predicted_classes)

    exit(0)

    for i in range(10):
        plt.imshow(kr.mnist_train_images[i], cmap=plt.cm.binary)
        plt.show()

    exit(0)
