# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from keras.models import Input, Model
from keras.layers import Dense
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
from nwae.utils.dataencoder.OneHotEncoder import OneHotEncoder


#
# Model to encode words to more compact vectors instead of the inefficient sparse one-hot encoding
# Каждое слово кодируется в более компактный n-мерный вектор с вещественными числами,
# а не в двоичном разреженном формате с только 1 и 0 как one-hot
#
# Суть решения
#   В принципе решение ниже очень просто, мы представим слово в вектор с аттрибутами.
#   Эти аттрибуты является самыми словами, и озаначается что мы должен быть только находить
#   связи между словами и создавать какую-то меру.
#
# Принцип алгоритма
#   В первом слое нейронной сети, входные данные (одно слово) кодируется как one-hot
#   В втором слое, кодирование выходных в n-мерном пространстве, который является вектор слова
#   В третом как обычно кодирование в softmax (вероятность)
#
class WordEmbedding:

    def __init__(
            self,
            identifier_string,
            lang,
            training_text_list,
            # If None, will not do spelling correction
            dir_path_model = None,
            # If None, will not replace any word with unknown symbol W_UNK
            model_features_list = None,
            dirpath_synonymlist = None,
            postfix_synonymlist = None,
            dir_wordlist = None,
            postfix_wordlist = None,
            dir_wordlist_app = None,
            postfix_wordlist_app = None,
            stopwords_list = None,
    ):
        self.identifier_string = identifier_string
        self.lang = lang
        self.training_text_list = training_text_list

        self.txt_pp = TxtPreprocessor(
            identifier_string = self.identifier_string,
            # If None, will not do spelling correction
            dir_path_model = dir_path_model,
            # If None, will not replace any word with unknown symbol W_UNK
            model_features_list = model_features_list,
            lang = self.lang,
            dirpath_synonymlist = dirpath_synonymlist,
            postfix_synonymlist = postfix_synonymlist,
            dir_wordlist = dir_wordlist,
            postfix_wordlist = postfix_wordlist,
            dir_wordlist_app = dir_wordlist_app,
            postfix_wordlist_app = postfix_wordlist_app,
            stopwords_list = stopwords_list,
        )

        self.index_word_dict = None
        self.word_index_dict = None
        return

    def preprocess_text(
            self
    ):
        self.sentences_cleaned = [
            self.txt_pp.process_text(inputtext=s, return_as_string=False)
            for s in self.training_text_list
        ]
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Processed sentences: ' + str(self.sentences_cleaned)
        )
        return self.sentences_cleaned

    #
    # Концепция заключается в том что "связь" между словом и "атрибутами" (сами является теми же словами)
    # измерится как расстояние слов
    #
    def form_word_tuples(
            self,
            window,
            sentences_array,
    ):
        assert window >= 2, 'Window ' + str(window) + ' must be >= 2.'

        # Creating a placeholder for the scanning of the word list
        word_lists = []
        all_text = []

        for sent in sentences_array:
            # Appending to the all text list
            all_text += sent

            # Creating a context dictionary
            for i, word in enumerate(sent):
                for w in range(window):
                    # Getting the context that is ahead by *window* words
                    if i + 1 + w < len(sent):
                        word_lists.append([word] + [sent[(i + 1 + w)]])
                    # Getting the context that is behind by *window* words
                    if i - w - 1 >= 0:
                        word_lists.append([word] + [sent[(i - w - 1)]])
        return word_lists, all_text

    #
    # Алгоритм обяснен выше
    #
    def encode(
            self,
            # E.g. {'china': 1, 'russia': 2, ..}
            word_list,
            # E.g. [('china', 'dimsum'), ('russia', 'xleb'), ..]
            word_tuples_list,
    ):
        oh_enc = OneHotEncoder()
        self.words_onehot = oh_enc.encode(
            feature_list = word_list
        )
        self.word_index_dict = oh_enc.get_feature_index_dict()
        self.index_word_dict = {v:k for k,v in self.word_index_dict.items()}
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique word dictionary, length ' + str(len(self.word_index_dict))
            + ': ' + str(self.word_index_dict)
        )

        X = []
        Y = []
        for t in word_tuples_list:
            root_word = t[0]
            root_word_index = self.word_index_dict[root_word]
            close_word = t[1]
            close_word_index = self.word_index_dict[close_word]
            X.append(self.words_onehot[root_word_index])
            Y.append(self.words_onehot[close_word_index])

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': X: ' + str(X) + '\nY: ' + str(Y)
        )

        return np.array(X), np.array(Y)

    def train(
            self,
            X,
            Y
    ):
        # Defining the size of the embedding
        embed_size = 2

        # Defining the neural network
        inp = Input(shape=(X.shape[1],))
        Log.debug('Input shape: ' + str(X.shape))
        # Middle layer is the embedding vector we seek to extract
        # "linear" because this will serve as the word definition, to be input to other neural networks
        x = Dense(units=embed_size, activation='linear')(inp)
        # Standard softmax final layer
        x = Dense(units=Y.shape[1], activation='softmax')(x)
        model = Model(inputs=inp, outputs=x)
        Log.debug('Output shape: ' + str(Y.shape))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()

        # Optimizing the network weights
        model.fit(
            x=X,
            y=Y,
            batch_size=256,
            epochs=100
        )

        # Obtaining the weights from the neural network.
        # These are the so called word embeddings

        # The input layer (embedding weights)
        weights = model.get_weights()[0]
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Weights extracted as embedding layer: ' + str(weights)
        )
        print(len(weights))

        # Creating a dictionary to store the embeddings in. The key is a unique word and
        # the value is the numeric vector
        embedding_dict = {}
        for word in self.word_index_dict.keys():
            embedding_dict.update({
                word: weights[self.word_index_dict.get(word)]
            })
        return embedding_dict


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    data_and_clean_version = [
        (
            'Аккаунт популярного южнокорейского чат-бота был заблокирован в Facebook после жалоб на ненавистнические '
            'высказывания в адрес сексуальных меньшинств.',
            # "Чистая" форма
            'Аккаунт популярный южнокорея чат-бот быть заблокировать в Facebook после жалоба на ненависть '
            'высказывание в адрес сексуальный меньшинство.'
         ),
        (
            'Как передаёт газета, в одном из диалогов с пользователями бот по имени Lee Ludа назвала лесбиянок '
            '«жуткими» и призналась, что ненавидит их.',
            # "Чистая" форма
            'Как передать газета, в один из диалог с пользователь бот по имя Lee Ludа назвать лесбиянка '
            '«жуткий» и признать, что ненавидеть они.'
        ),
        (
            'По словам издания, это уже не первый случай, когда искусственный интеллект сталкивается с обвинениями '
            'в нетерпимости и дискриминации.',
            # "Чистая" форма
            'По слово издание, это уже не первый случай, когда искусственный интеллект сталкиваться с обвинение '
            'в нетерпимость и дискриминация.'
        ),
        (
            'Чат-бот Lee Ludа, разработанный сеульским стартапом Scatter Lab и использовавший личность 20-летней '
            'студентки университета, был удалён из Facebook messenger на этой неделе',
            # "Чистая" форма
            'Чат-бот Lee Ludа, разработать сеул стартап Scatter Lab и использовать личность 20-лет '
            'студентка университет, быть удалить из Facebook messenger на это неделя'
        ),
        (
            'Luda поразила пользователей глубиной и естественным тоном своих ответов, которые она заимствовала '
            'из 10 млрд реальных разговоров между молодыми парами в самом популярном южнокорейском мессенджере KakaoTalk',
            # "Чистая" форма
            'Luda поразить пользователь глубиний и естественный тон свой ответ, который она заимствовать '
            'из 10 млрд реальный разговор между молодой пара в самый популярный южнокорея мессенджер KakaoTalk'
        ),
        (
            'восхищение знанием интернет-сленга переросло в возмущение после того, как Luda начала использовать '
            'оскорбительные и откровенно сексуальные термины',
            # "Чистая" форма
            'восхищение знание интернет-сленг перерасти в возмущение после тот, как Luda начать использовать '
            'оскорбительный и откровенный сексуальный термин'
        ),
    ]

    from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
    stopwords = [
        'как', 'на', 'в', 'и', 'с', 'по', 'не', 'когда', 'который',
        'из', 'что', 'это', 'тот',
        'он', 'она', 'оно', 'они', 'свой',
        'быть', 'уже', 'после', 'между',
        'один',
    ] + [p for p in BasicPreprocessor.DEFAULT_PUNCTUATIONS]
    stopwords = [s.lower() for s in stopwords]
    print('Using stopwords: ' + str(stopwords))

    we = WordEmbedding(
        identifier_string = 'test.word.embedding',
        lang = 'ru',
        training_text_list = [s[1] for s in data_and_clean_version],
        stopwords_list = stopwords,
    )
    sentences = we.preprocess_text()
    special_symbols = [BasicPreprocessor.W_UNK, BasicPreprocessor.W_NUM]
    cleaned_sentences = []
    for sent in sentences:
        cleaned_sentences.append(
            [w for w in sent if w not in special_symbols]
        )
    [print(s) for s in cleaned_sentences]

    word_tuples_list, all_text = we.form_word_tuples(
        window = 2,
        sentences_array = cleaned_sentences,
    )
    print('Word tuples list: ' + str(word_tuples_list))
    print('All text: ' + str(all_text))

    X, Y = we.encode(
        word_list = all_text,
        word_tuples_list = word_tuples_list
    )
    index_word_dict = we.index_word_dict
    v = np.array(list(range(len(X[0]))))

    # Check the one-hot encoding
    for i, word_oh in enumerate(X):
        root_word_index = np.sum(v*word_oh)
        close_word_index = np.sum(v*Y[i])
        root_word = index_word_dict[root_word_index]
        close_word = index_word_dict[close_word_index]
        pair = [root_word, close_word]
        assert pair in word_tuples_list

    embedding_dict = we.train(X=X, Y=Y)
    print(embedding_dict)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for word in list(we.word_index_dict.keys()):
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
    plt.show()
    exit(0)
