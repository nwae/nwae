# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.ml.ModelInterface import ModelInterface

try:
    from keras import models
    from keras import layers
    from keras.utils import to_categorical
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing Keras modules: ' + str(ex_keras)
    )


class NetworkDesign:

    KEY_MODEL_TYPE = 'model_type'
    KEY_MODEL_LAYERS = 'model_layers'
    # For some architecture, we must not convert to_categorical() the labels for network fitting
    KEY_MODEL_LABEL_TO_CATEGORICAL = 'label_require_to_categorical'

    MODEL_GENERAL_DATA   = 'general_data_2_layer'
    MODEL_TEXT_EMBEDDING = 'text_model_with_embedding_layer'

    def __init__(
            self,
            # Если вид модели пустой/None, network_layer_config обязанно существует
            model_type              = None,
            max_label_value         = None,
            # For general data model
            input_shape             = None,
            # For text embedding model
            txtemb_max_sentence_len = None,
            txtemb_vocabulary_size  = None,
            # Ручный дисайн
            network_layer_config    = None,
    ):
        self.model_type = model_type
        self.max_label_value = max_label_value
        self.input_shape = input_shape
        self.txtemb_max_sentence_len = txtemb_max_sentence_len
        self.txtemb_vocabulary_size = txtemb_vocabulary_size
        self.network_layer_config = network_layer_config

        if self.network_layer_config is None:
            # TensorFlow will fail with only 1 label
            if self.max_label_value <= 1:
                self.max_label_value = 2

        self.require_label_to_categorical = True
        self.network = None
        return

    def summary(self):
        return self.network.summary()

    def get_network_config(self):
        return self.network_layer_config

    def get_network(self):
        return self.network

    def create_network(
            self
    ):
        self.design_network_layers()
        self.create_network_from_layer_config()
        return

    def create_network_from_layer_config(
            self,
    ):
        try:
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': Start creating network layers from config: ' + str(self.network_layer_config)
                + '. Config created from model type "' + str(self.model_type)
                + ', input shape ' + str(self.input_shape)
                + ', max label value = ' + str(self.max_label_value)
                + ', max sentence len = ' + str(self.txtemb_max_sentence_len)
                + ', vocabulary size = ' + str(self.txtemb_vocabulary_size)
            )
            self.require_label_to_categorical = \
                self.network_layer_config[NetworkDesign.KEY_MODEL_LABEL_TO_CATEGORICAL]

            self.network = models.Sequential()

            layers_config = self.network_layer_config[NetworkDesign.KEY_MODEL_LAYERS]
            for i_layer in range(len(layers_config)):
                ly = layers_config[i_layer]
                layer_type = ly[ModelInterface.NN_LAYER_TYPE]
                if layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_DENSE:
                    if i_layer == 0:
                        self.network.add(
                            layers.Dense(
                                units       = ly[ModelInterface.NN_LAYER_OUTPUT_UNITS],
                                activation  = ly[ModelInterface.NN_LAYER_ACTIVATION],
                                input_shape = ly[ModelInterface.NN_LAYER_INPUT_SHAPE]
                            )
                        )
                    else:
                        self.network.add(
                            layers.Dense(
                                units      = ly[ModelInterface.NN_LAYER_OUTPUT_UNITS],
                                activation = ly[ModelInterface.NN_LAYER_ACTIVATION]
                            )
                        )
                elif layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_EMBEDDING:
                    # Embedding layers are not part of the NN optimization, its only function
                    # is to transform the input to a standard NN form
                    self.network.add(
                        # Embedding layers are usually for text data
                        layers.embeddings.Embedding(
                            # Must be at least the size of the unique vocabulary
                            input_dim    = ly[ModelInterface.NN_LAYER_INPUT_DIM],
                            # How many words, or length of word input vector
                            input_length = ly[ModelInterface.NN_LAYER_INPUT_LEN],
                            # Each word represented by a vector of dimension 8 (for example)
                            output_dim   = ly[ModelInterface.NN_LAYER_OUTPUT_DIM]
                        )
                    )
                elif layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_FLATTEN:
                    # Usually comes after an embedding layer, does nothing but to convert
                    # a 2 dimensional input to a 1-dimensional output by the usual "flattening"
                    self.network.add(layers.Flatten())
                elif layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_DROPOUT:
                    self.network.add(
                        layers.Dropout(rate = ly[ModelInterface.NN_LAYER_DROPOUT_RATE])
                    )
                elif layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_CONV2D:
                    if i_layer == 0:
                        self.network.add(
                            layers.Conv2D(
                                filters     = ly[ModelInterface.NN_LAYER_CONV_FILTERS],
                                kernel_size = ly[ModelInterface.NN_LAYER_CONV_KERNEL_SIZE],
                                input_shape = ly[ModelInterface.NN_LAYER_INPUT_SHAPE],
                                activation  = ly[ModelInterface.NN_LAYER_ACTIVATION]
                            )
                        )
                    else:
                        self.network.add(
                            layers.Conv2D(
                                filters     = ly[ModelInterface.NN_LAYER_CONV_FILTERS],
                                kernel_size = ly[ModelInterface.NN_LAYER_CONV_KERNEL_SIZE],
                                activation  = ly[ModelInterface.NN_LAYER_ACTIVATION]
                            )
                        )
                elif layer_type == ModelInterface.VALUE_NN_LAYER_TYPE_MAXPOOL2D:
                    self.network.add(
                        layers.MaxPooling2D(pool_size=ly[ModelInterface.NN_LAYER_POOLING_SIZE])
                    )
                else:
                    raise Exception('Unrecognized layer type "' + str(layer_type) + '"')
            Log.info(
                str(self.__class__) + str(getframeinfo(currentframe()).lineno)
                + ': Successfully created network layers from config: ' + str(self.network_layer_config)
            )
        except Exception as ex_layers:
            self.network = None
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error creating network layers for config: ' + str(self.network_layer_config) \
                     + '. Exception: ' + str(ex_layers)
            Log.error(errmsg)
            raise Exception(errmsg)

    def design_network_layers(
            self
    ):
        last_layer_output_len = self.max_label_value + 1

        if self.model_type == NetworkDesign.MODEL_TEXT_EMBEDDING:
            if self.txtemb_vocabulary_size < 2**8:
                output_dim = 8
            else:
                output_dim = 16

            self.network_layer_config = self.__design_text_embedding_model(
                middle_layer_output_len    = last_layer_output_len * 5,
                last_layer_output_len      = last_layer_output_len,
                embedding_layer_output_dim = output_dim,
                embedding_layer_input_len  = self.txtemb_max_sentence_len,
                embedding_layer_input_dim  = self.txtemb_vocabulary_size
            )
        elif self.model_type == NetworkDesign.MODEL_GENERAL_DATA:
            self.network_layer_config =  self.__design_general_data_model(
                last_layer_output_len = last_layer_output_len
            )
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Model type "' + str(self.model_type) + '" not supported!'
            )
        return self.network_layer_config

    def __design_general_data_model(
            self,
            last_layer_output_len
    ):
        return {
            NetworkDesign.KEY_MODEL_TYPE: self.model_type,
            NetworkDesign.KEY_MODEL_LABEL_TO_CATEGORICAL: True,
            NetworkDesign.KEY_MODEL_LAYERS: [
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                    ModelInterface.NN_LAYER_OUTPUT_UNITS: last_layer_output_len * 5,
                    # First layer just makes sure to output positive numbers with linear rectifier
                    ModelInterface.NN_LAYER_ACTIVATION: 'relu',
                    ModelInterface.NN_LAYER_INPUT_SHAPE: self.input_shape
                },
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                    ModelInterface.NN_LAYER_OUTPUT_UNITS: last_layer_output_len,
                    # Last layer uses a probability based output softmax
                    ModelInterface.NN_LAYER_ACTIVATION: 'softmax'
                }
            ]
        }

    def __design_text_embedding_model(
            self,
            # Ad-hoc, but we can take 2 or 3 times last layer length
            middle_layer_output_len,
            # Usually how many labels we have or max label value + 1
            last_layer_output_len,
            # 16 dim (2^16=65536) should be more than enough to cover our vocabulary
            embedding_layer_output_dim,
            # Usually max sentence length
            embedding_layer_input_len,
            # Usually how many unique vocabulary words
            embedding_layer_input_dim,
    ):
        return {
            NetworkDesign.KEY_MODEL_TYPE: self.model_type,
            NetworkDesign.KEY_MODEL_LABEL_TO_CATEGORICAL: False,
            NetworkDesign.KEY_MODEL_LAYERS: [
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_EMBEDDING,
                    # 16 dim (2^16=65536) should be more than enough to cover our vocabulary
                    ModelInterface.NN_LAYER_OUTPUT_DIM: embedding_layer_output_dim,
                    # Usually max sentence length
                    ModelInterface.NN_LAYER_INPUT_LEN: embedding_layer_input_len,
                    # Usually how many unique vocabulary words
                    ModelInterface.NN_LAYER_INPUT_DIM: embedding_layer_input_dim
                },
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_FLATTEN
                },
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                    ModelInterface.NN_LAYER_OUTPUT_UNITS: middle_layer_output_len,
                    # First layer just makes sure to output positive numbers with linear rectifier
                    ModelInterface.NN_LAYER_ACTIVATION: 'relu',
                },
                {
                    ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                    ModelInterface.NN_LAYER_OUTPUT_UNITS: last_layer_output_len,
                    # Last layer uses a probability based output softmax
                    ModelInterface.NN_LAYER_ACTIVATION: 'softmax'
                }
            ]
        }


if __name__ == '__main__':
    model_design = NetworkDesign(
        model_type      = NetworkDesign.MODEL_TEXT_EMBEDDING,
        max_label_value = 2651,
        txtemb_max_sentence_len = 12,
        txtemb_vocabulary_size  = 74
    )
    model_design.create_network()
    model_design.summary()

    # Allow to create with manual layer design
    nn_layers = {
        NetworkDesign.KEY_MODEL_TYPE: NetworkDesign.MODEL_GENERAL_DATA,
        NetworkDesign.KEY_MODEL_LABEL_TO_CATEGORICAL: True,
        NetworkDesign.KEY_MODEL_LAYERS: [
            {
                ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                ModelInterface.NN_LAYER_OUTPUT_UNITS: 512,
                # First layer just makes sure to output positive numbers with linear rectifier
                ModelInterface.NN_LAYER_ACTIVATION:   'relu',
                ModelInterface.NN_LAYER_INPUT_SHAPE:  (888,)
            },
            {
                ModelInterface.NN_LAYER_TYPE: ModelInterface.VALUE_NN_LAYER_TYPE_DENSE,
                ModelInterface.NN_LAYER_OUTPUT_UNITS: 20,
                # Last layer uses a probability based output softmax
                ModelInterface.NN_LAYER_ACTIVATION:   'softmax'
            }
        ]
    }
    model_design_2 = NetworkDesign(
        network_layer_config = nn_layers
    )
    model_design_2.create_network_from_layer_config()
    model_design_2.summary()

