# -*- coding: utf-8 -*-

import numpy as np
from nwae.lib.math.NumpyUtil import NumpyUtil
import keras.layers as keraslay
import keras.utils as kerasutils
from keras.models import Sequential
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.utils.FileUtils import FileUtils


# Training data or Documents
# seq_pairs = [
#     ('Я иду на работу', '나는 일에게 간다'),
#     ('Я люблю мою работу', '나는 내 일을 사랑해'),
#     ('Моя работа интересная', '나의 일은 흥미홉다')
# ]

def get_sample_lang_pairs(
        filepath
):
    flines = FileUtils.read_text_file(filepath=filepath)
    i=0
    lang_pairs = []
    for line in flines:
        try:
            arr = line.split('\t')
            pair = [arr[0], arr[1]]
            lang_pairs.append(pair)
            i+=1
        except Exception as ex:
            errmsg = 'Cannot split line ' + str(i) + ' "' + str(line) + '"'
            print(errmsg)
    return lang_pairs


def create_padded_doc_pairs(
        docs_pairs
):
    import keras.preprocessing as kerasprep

    cleaned_sentences = {}
    indexed_dicts = {}
    idx_sentences = {}
    max_len = {}
    for i in range(2):
        sentences = [x[i].split(' ') for x in docs_pairs]

        cleaned_sentences[i] = [ BasicPreprocessor.clean_punctuations(sentence=sent) for sent in sentences ]
        #print('Cleaned Sentences ' + str(i) + ': ' + str(cleaned_sentences[i]))

        indexed_dicts[i] = BasicPreprocessor.create_indexed_dictionary(
            sentences = cleaned_sentences[i]
        )
        #print('Indexed Dict ' + str(i) + ': ' + str(indexed_dicts[i]))

        idx_sentences[i] = BasicPreprocessor.sentences_to_indexes(
            sentences    = cleaned_sentences[i],
            indexed_dict = indexed_dicts[i]
        )
        #print('Indexed Sentences ' + str(i) + ': ' + str(idx_sentences[i]))

        max_len[i] = BasicPreprocessor.extract_max_length(
            corpora = idx_sentences[i]
        )

    data_set = BasicPreprocessor.prepare_sentence_pairs(
        sentences_l1 = idx_sentences[0],
        sentences_l2 = idx_sentences[1],
        len_l1       = max_len[0],
        len_l2       = max_len[1]
    )

    class RetClass:
        def __init__(self, data_set, cleaned_sentences, indexed_dicts, idx_sentences, max_len):
            self.data_set = data_set
            self.cleaned_sentences = cleaned_sentences
            self.indexed_dicts = indexed_dicts
            self.idx_sentences = idx_sentences
            self.max_len = max_len

    # print('Data set: ' + str(data_set))
    return RetClass(
        data_set          = data_set,
        cleaned_sentences = cleaned_sentences,
        indexed_dicts     = indexed_dicts,
        idx_sentences     = idx_sentences,
        max_len           = max_len
    )


#
# The neural network model training
#
def create_seq_text_model(
        embedding_input_dim,
        embedding_output_dim,
        embedding_input_length,
        class_labels,
        lstm_units_multiplier,
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
        model.add(keraslay.LSTM(
            units = unique_labels_count*lstm_units_multiplier
        ))
        # model.add(keraslay.Dense(units=unique_labels_count*mid_layer_units_multiplier, activation='relu'))
        model.add(keraslay.Dense(units=unique_labels_count, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Train/Fit the model. Don't know why need to use 'sparse_categorical_crossentropy'
        # and labels instead of labels_categorical

    # summarize the model
    print(model.summary())
    return model


lang_pairs = get_sample_lang_pairs(filepath='/usr/local/git/nwae/nwae/nlp.data/perevod/en.ko.txt')
print(lang_pairs[100:120])

ret = create_padded_doc_pairs(
    docs_pairs = lang_pairs
)
cleaned_sents = ret.cleaned_sentences
indexed_dicts = ret.indexed_dicts
print(indexed_dicts)
idx_sents = ret.idx_sentences
print(idx_sents[0][100:120])
print(idx_sents[1][100:120])
maxlen = ret.max_len
print(maxlen)
exit(0)

model_text = create_seq_text_model(
    embedding_input_dim  = ret.vocabulary_dimension,
    embedding_output_dim = 8,
    embedding_input_length = ret.input_dim_max_length,
    class_labels         = ret.list_labels,
    lstm_units_multiplier = 2
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
