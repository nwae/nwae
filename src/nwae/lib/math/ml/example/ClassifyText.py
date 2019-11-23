# -*- coding: utf-8 -*-

#
# From https://learn-neural-networks.com/world-embedding-by-keras/
#

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


# Training data or Documents
docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.'
]
# Class labels
lbls = array([1,1,1,1,1,0,0,0,0,0])

#
# Process sentences into numbers, with padding
#
# Vocabulary dimension
vs = 50
enc_docs = [one_hot(d, vs) for d in docs]
print(enc_docs)

# pad documents to a max length of 4 words
max_length = 4
p_docs = pad_sequences(enc_docs, maxlen=max_length, padding='post')
print(p_docs)

#
# The neural network model training
#
# define the model
modelEmb = Sequential()
modelEmb.add(
    Embedding(
        input_dim    = vs,
        output_dim   = 8,
        input_length = max_length
    )
)
modelEmb.add(Flatten())
modelEmb.add(Dense(1, activation='sigmoid'))
# compile the model
modelEmb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(modelEmb.summary())
# fit the model
modelEmb.fit(p_docs, lbls, epochs=150, verbose=0)

#
# Evaluate the model
#
loss, accuracy = modelEmb.evaluate(p_docs, lbls, verbose=2)
print('Accuracy: %f' % (accuracy*100))