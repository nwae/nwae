# -*- coding: utf-8 -*-

#
# Partly from https://learn-neural-networks.com/world-embedding-by-keras/
#

from numpy import array
import keras.preprocessing as kerasprep
import keras.layers as keraslay
from keras.models import Sequential
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor


# Training data or Documents
docs_label = [
    ('잘 했어!',1), ('잘 했어요!',1), ('잘 한다!',1),
    ('Молодец!',1), ('Супер!',1), ('Хорошо!',1),
    ('Плохо!',0), ('Дурак!',0),
    ('나쁜!',0), ('바보!',0), ('백치!',0), ('얼간이!',0),
    ('미친놈',0), ('씨발',0), ('개',0), ('개자식',0),
    ('젠장',0)
]
docs = [x[0].split(' ') for x in docs_label]
docs = [BasicPreprocessor.clean_punctuations(sentence=sent) for sent in docs]
labels = [x[1] for x in docs_label]
print('Docs: ' + str(docs))
print('Labels: ' + str(labels))

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
for sent_label in docs_label:
    max_length = max(len(sent_label[0].split(' ')), max_length)
print('Max Length = ' + str(max_length))

p_docs = kerasprep.sequence.pad_sequences(enc_docs, maxlen=max_length, padding='post')
print('Padded Encoded Sentences (' + str(len(p_docs)) + '):')
print(p_docs)

#
# The neural network model training
#
# define the model
modelEmb = Sequential()
#
# If each sentence has n words, and each word is 8 dimensions (output dim
# of embedding layer), this means the final output of a sentence is (n,8)
# in dimension.
#
embedding_layer = keraslay.embeddings.Embedding(
        input_dim    = vs,
        # Each word represented by a vector of dimension 8
        output_dim   = 8,
        input_length = max_length
    )

# Model
modelEmb.add(embedding_layer)
# After flattening each sentence of shape (4,8), will have shape (32)
modelEmb.add(keraslay.Flatten())
# Our standard dense layer, with 1 node
modelEmb.add(keraslay.Dense(1, activation='sigmoid'))
# compile the model
modelEmb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(modelEmb.summary())
# fit the model
modelEmb.fit(p_docs, array(labels), epochs=150, verbose=0)

#
# Evaluate the model
#
loss, accuracy = modelEmb.evaluate(p_docs, array(labels), verbose=2)
print('Accuracy: %f' % (accuracy*100))

probs = modelEmb.predict(p_docs)
print('Probs:' + str(probs))
