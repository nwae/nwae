# -*- coding: utf-8 -*-

import numpy as np
from nwae.math.NumpyUtil import NumpyUtil
import keras.layers as keraslay
from keras.models import Sequential
from nwae.ml.text.TxtTransform import TxtTransform


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

res = TxtTransform(
    docs=[x[0] for x in docs_label],
    labels=[x[1] for x in docs_label]
).create_padded_docs()

model_text = create_text_model(
    embedding_input_dim  = res.vocabulary_dimension,
    embedding_output_dim = 8,
    embedding_input_length = res.max_x_length,
    class_labels         = res.encoded_labels,
    binary               = False
)
padded_encoded_docs = res.padded_encoded_docs
encoded_labels = res.encoded_labels
encoded_labels_cat = res.encoded_labels_categorical
print('Padded docs: ' + str(padded_encoded_docs))
print('List encoded labels: ' + str(encoded_labels_cat))
model_text.fit(padded_encoded_docs, np.array(encoded_labels), epochs=150, verbose=0)

#
# Evaluate the model
#
loss, accuracy = model_text.evaluate(padded_encoded_docs, np.array(encoded_labels), verbose=2)
print('Accuracy: %f' % (accuracy*100))

probs = model_text.predict(padded_encoded_docs)
print('Probs:' + str(probs))

# Compare some data
count_correct = 0
for i in range(len(padded_encoded_docs)):
    data_i = np.array([padded_encoded_docs[i]])
    label_i = encoded_labels[i]
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
print('Accuracy = ' + str(100*count_correct/len(padded_encoded_docs)) + '%.')
