# -*- coding: utf-8 -*-

#
# Partly from https://learn-neural-networks.com/world-embedding-by-keras/
#

import numpy as np
from nwae.lib.math.NumpyUtil import NumpyUtil
import keras.layers as keraslay
import keras.utils as kerasutils
from keras.models import Sequential
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor


# Training data or Documents
docs_label = [
    ('잘 했어!',1), ('잘 했어요!',1), ('잘 한다!',1),
    ('Молодец!',1), ('Супер!',1), ('Хорошо!',1),
    ('Плохо!',0), ('Дурак!',0),
    ('나쁜!',0), ('바보!',0), ('백치!',0), ('얼간이!',0),
    ('미친놈',0), ('씨발',0), ('개',0), ('개자식',0),
    ('젠장',0),
    ('ok',2), ('fine',2)
]

def create_padded_docs(
        list_docs_label
):
    import keras.preprocessing as kerasprep

    docs = [x[0].split(' ') for x in list_docs_label]
    docs = [BasicPreprocessor.clean_punctuations(sentence=sent) for sent in docs]
    labels = [x[1] for x in list_docs_label]
    # How many unique labels
    n_labels = len(list(set(labels)))
    # Convert labels to categorical one-hot encoding
    labels_categorical = kerasutils.to_categorical(labels, num_classes=n_labels)

    print('Docs: ' + str(docs))
    print('Labels: ' + str(labels))
    print('Labels converted to categorical: ' + str(labels_categorical))

    unique_words = list(set([w for sent in docs for w in sent]))
    print('Unique words: ' + str(unique_words))

    #
    # Create indexed dictionary
    #
    one_hot_dict = BasicPreprocessor.create_indexed_dictionary(
        sentences = docs
    )
    print('One Hot Dict: ' + str(one_hot_dict))

    #
    # Process sentences into numbers, with padding
    # In real environments, we usually also replace unknown words, numbers, URI, etc.
    # with standard symbols, do word stemming, remove stopwords, etc.
    #
    # Vocabulary dimension
    vs = len(unique_words) + 10
    enc_docs = BasicPreprocessor.sentences_to_indexes(
        sentences = docs,
        indexed_dict = one_hot_dict
    )
    print('Encoded Sentences (' + str(len(enc_docs)) + '):')
    print(enc_docs)

    # pad documents to a max length of 4 words
    max_length = 1
    for sent in enc_docs:
        max_length = max(len(sent), max_length)
    print('Max Length = ' + str(max_length))

    p_docs = kerasprep.sequence.pad_sequences(enc_docs, maxlen=max_length, padding='pre')
    print('Padded Encoded Sentences (' + str(p_docs.shape) + '):')
    print(p_docs)

    class RetClass:
        def __init__(self, encoded_docs, padded_docs, list_labels, list_labels_categorical, vocabulary_dimension, input_dim_max_length):
            self.encoded_docs = encoded_docs
            self.padded_docs = padded_docs
            self.list_labels = list_labels
            self.list_labels_categorical = list_labels_categorical
            self.vocabulary_dimension = vocabulary_dimension
            self.input_dim_max_length = input_dim_max_length

    return RetClass(
        encoded_docs = enc_docs,
        padded_docs  = p_docs,
        list_labels = labels,
        list_labels_categorical = labels_categorical,
        vocabulary_dimension = vs,
        input_dim_max_length = max_length
    )

#
# The neural network model training
#
def create_text_model(
        embedding_input_dim,
        embedding_output_dim,
        embedding_input_length,
        class_labels,
        mid_layer_units_multiplier = 5,
        binary = False
):
    unique_labels_count = len(list(set(class_labels)))

    # define the model
    model = Sequential()
    #
    # If each sentence has n words, and each word is 8 dimensions (output dim
    # of embedding layer), this means the final output of a sentence is (n,8)
    # in dimension.
    #
    embedding_layer = keraslay.embeddings.Embedding(
        input_dim    = embedding_input_dim,
        # Each word represented by a vector of dimension 8
        output_dim   = embedding_output_dim,
        # How many words, or length of input vector
        input_length = embedding_input_length
    )

    # Model
    model.add(embedding_layer)
    # After flattening each sentence of shape (max_length,8), will have shape (max_length*8)
    model.add(keraslay.Flatten())

    if binary:
        # Our standard dense layer, with 1 node
        model.add(keraslay.Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    else:
        # Accuracy drops using 'sigmoid'
        model.add(keraslay.Dense(units=unique_labels_count*mid_layer_units_multiplier, activation='relu'))
        model.add(keraslay.Dense(units=unique_labels_count, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Train/Fit the model. Don't know why need to use 'sparse_categorical_crossentropy'
        # and labels instead of labels_categorical

    # summarize the model
    print(model.summary())
    return model

ret = create_padded_docs(
    list_docs_label = docs_label
)

model_text = create_text_model(
    embedding_input_dim  = ret.vocabulary_dimension,
    embedding_output_dim = 8,
    embedding_input_length = ret.input_dim_max_length,
    class_labels         = ret.list_labels,
    binary               = False
)
model_text.fit(ret.padded_docs, np.array(ret.list_labels), epochs=150, verbose=0)

#
# Evaluate the model
#
loss, accuracy = model_text.evaluate(ret.padded_docs, np.array(ret.list_labels), verbose=2)
print('Accuracy: %f' % (accuracy*100))

probs = model_text.predict(ret.padded_docs)
print('Probs:' + str(probs))

# Compare some data
count_correct = 0
for i in range(len(ret.padded_docs)):
    data_i = np.array([ret.padded_docs[i]])
    label_i = ret.list_labels[i]
    prob_distribution = model_text.predict(x=data_i)
    print('Model prob distribution from softmax: ' + str(prob_distribution))
    top_x = NumpyUtil.get_top_indexes(
        data      = prob_distribution[0],
        ascending = False,
        top_x     = 5
    )
    if top_x[0] == label_i:
        count_correct += 1
    print(str(i) + '. ' + str(data_i) + ': Label=' + str(label_i) + ', predicted=' + str(top_x))
print('Accuracy = ' + str(100*count_correct/len(ret.padded_docs)) + '%.')
