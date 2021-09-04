# -*- coding: utf-8 -*-

import numpy as np
import keras.utils as kerasutils
import keras.layers as keraslay
from keras.models import Sequential
from nwae.lib.math.NumpyUtil import NumpyUtil

#
# Prepare random data
#
n_rows = 10000
input_dim = 5
n_labels = input_dim
# Random vectors numpy ndarray type
data = np.random.random((n_rows, input_dim))
# Labels are sum of the rows, then floored to the integer
# Sum >= 0, 1, 2, 3,...
row_sums = np.sum(data, axis=1)
labels = np.array(np.round(row_sums-0.5,0), dtype=int)
print(labels)
# labels = np.random.randint(n_labels, size=(n_rows, 1))

# Print some data
for i in range(10):
    print(str(i) + '. ' + str(labels[i]) + ': ' + str(data[i]))

#
# Build our NN
#
def create_general_model(
        input_data_dim,
        unique_labels_count,
        dense_units
):
    model = Sequential()
    # First layer with standard relu or positive activation
    model.add(
        keraslay.Dense(
            # Output dim somewhat ad-hoc
            units      = dense_units,
            activation = 'relu',
            input_dim  = input_data_dim
        )
    )
    # Subsequent layers input dim no longer required to be specified, implied from previous
    # Last layer always outputs the labels probability/scores
    model.add(
        keraslay.Dense(
            # Output dim
            units      = unique_labels_count,
            # Standard last layer activation as positive probability distribution
            activation = 'softmax'
        )
    )
    model.compile(
        optimizer = 'rmsprop',
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )
    model.summary()
    return model

model_general = create_general_model(
    input_data_dim = input_dim,
    unique_labels_count = n_labels,
    dense_units  = 512
)
# Convert labels to categorical one-hot encoding
labels_categorical = kerasutils.to_categorical(labels, num_classes=n_labels)

# Train the model, iterating on the data in batches of 32 samples
model_general.fit(data, labels_categorical, epochs=10, batch_size=32)

loss, accuracy = model_general.evaluate(data, labels_categorical)
print('Accuracy: %f' % (accuracy*100))

# Compare some data
count_correct = 0
for i in range(data.shape[0]):
    data_i = np.array([data[i]])
    label_i = labels[i]
    prob_distribution = model_general.predict(x=data_i)
    top_x = NumpyUtil.get_top_indexes(
        data      = prob_distribution[0],
        ascending = False,
        top_x     = 5
    )
    if top_x[0] == label_i:
        count_correct += 1
    print(str(i) + '. ' + str(data_i) + ': Label=' + str(label_i) + ', predicted=' + str(top_x))
print('Accuracy = ' + str(100*count_correct/data.shape[0]) + '%.')
