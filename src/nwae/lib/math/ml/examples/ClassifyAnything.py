# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import keras.utils as kerasutils
import keras.layers as keraslay
from keras.models import Sequential

#
# Prepare random data
#
n_rows = 1000
input_dim = 3
n_labels = 8
# Random vectors numpy ndarray type
data = np.random.random((n_rows, input_dim))
# Random labels
labels = np.random.randint(n_labels, size=(n_rows, 1))

# Print some data
for i in range(10):
    print(str(i) + '. ' + str(labels[i]) + ': ' + str(data[i]))

#
# Build our NN
#
model = Sequential()
# First layer with standard relu or positive activation
model.add(
    keraslay.Dense(
        # Output dim somewhat ad-hoc
        units      = 32,
        activation = 'relu',
        input_dim  = input_dim
    )
)
# Subsequent layers input dim no longer required to be specified, implied from previous
# Last layer always outputs the labels probability/scores
model.add(
    keraslay.Dense(
        # Output dim
        units      = n_labels,
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

# Convert labels to categorical one-hot encoding
one_hot_labels = kerasutils.to_categorical(labels, num_classes=n_labels)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(data, one_hot_labels)
print('Accuracy: %f' % (accuracy*100))

def get_top_x(
        # numpy ndarray shape (1, n_labels)
        prob_distribution,
        top_x = 5
):
    df = pd.DataFrame(data=prob_distribution, columns=['prob'])
    df.sort_values(by=['prob'], ascending=False, inplace=True)
    df['label'] = df.index
    #print(df)
    top_x = np.array(df.index[0:top_x])
    #print(top_x)
    return top_x


# Compare some data
for i in range(10):
    data_i = np.array([data[i]])
    label_i = labels[i]
    prob_distribution = model.predict(x=data_i)
    top_x = get_top_x(prob_distribution=prob_distribution[0])
    print(str(i) + '. ' + str(data_i) + ': Label=' + str(label_i) + ', predicted=' + str(top_x))
