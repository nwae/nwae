# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nagisa
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
import numpy as np
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lang.corpora.Scrape import Scrape


#
# TODO
#    Обобщи код для любого языка
#
class WordSegmentationModel:

    HIDDEN_STATE_NOT_WORD_SEPARATOR = 0
    HIDDEN_STATE_IS_WORD_SEPARATOR  = 1

    @staticmethod
    def get_training_data_from_file(
            filepath,
            min_char_per_sent = 10,
            max_char_per_sent = 30,
    ):
        f = open(file=filepath, mode='r', encoding='utf-8')
        sentences_list = f.readlines()
        sentences_list = [
            s for s in sentences_list
            if (len(s) >= min_char_per_sent) and (len(s) <= max_char_per_sent)
        ]
        return sentences_list

    @staticmethod
    def get_training_data_by_scraping_urls(
            url_list,
            tag_to_find       = 'p',
            min_char_per_sent = 10,
            max_char_per_sent = 30,
            write_to_filepath = None,
    ):
        sentences_list_agg = []
        for url in url_list:
            sentences_list = WordSegmentationModel.get_training_data_by_scraping(
                url = url,
                tag_to_find = tag_to_find,
                min_char_per_sent = min_char_per_sent,
                max_char_per_sent = max_char_per_sent,
            )
            sentences_list_agg += sentences_list
            #tokens_list_agg += tokens_list
            #is_sep_list_agg += is_sep_list

        if write_to_filepath:
            f = open(file=str(write_to_filepath), mode='w', encoding='utf-8')
            [ f.write(str(s) + '\n') for s in sentences_list_agg ]
            f.close()

        return sentences_list_agg

    @staticmethod
    def get_training_data_by_scraping(
            url,
            tag_to_find = 'p',
            min_char_per_sent = 10,
            max_char_per_sent = 30,
    ):
        # Пример данных из википедии
        sentences_list_from_wiki_scraping = Scrape().scrape_url(
            url = url,
            tag_to_find = tag_to_find
        )
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Scraped ' + str(len(sentences_list_from_wiki_scraping)) + ' sentences from url "' + str(url) + '"'
        )
        # Only take length 10-20
        sentences_list = [
            s for s in sentences_list_from_wiki_scraping
            if ((len(s) >= min_char_per_sent) and (len(s) <= max_char_per_sent))
        ]
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Filtered to ' + str(len(sentences_list)) + ' sentences from url "' + str(url) + '"'
        )
        return sentences_list

    @staticmethod
    def extract_tokens(sentences_list):
        tokens_tags_list, tokens_list, tags_list = WordSegmentationModel.tokenize_jp(
            sentences_list=sentences_list
        )
        is_sep_list = []
        for i in range(len(sentences_list)):
            s = sentences_list[i]
            split_tokens = tokens_list[i]
            is_sep = [0] * len(s)
            j_cum = 0
            for tok in split_tokens:
                j_cum += len(tok)
                is_sep[j_cum - 1] = 1
            is_sep_list.append(is_sep)
            Log.debug('Sentence "' + str(s) + '"')
            Log.debug('Tokenized "' + str(split_tokens) + '"')
            Log.debug('Is Sep: ' + str(is_sep))

        return tokens_list, is_sep_list

    def __init__(
            self,
            sentences_list,
            is_separators_list,
            embedding_vector_len = 2,
    ):
        self.sentences = sentences_list
        self.is_separators_list = is_separators_list
        self.embedding_vector_len = embedding_vector_len

        # Задние и передные буквы для подсказывания наличия или отсутствия пробела
        self.len_chr_lookback = 2
        self.len_chr_lookfwd = 2

        for i in range(len(self.sentences)):
            sent = self.sentences[i]
            seps = self.is_separators_list[i]
            assert len(sent) == len(seps), 'Sentence ' + str(sent) + ', seps ' + str(seps) + ' must be same len'
        self.__preprocess()
        return

    def get_sequence_and_prediction_x_y(
            self,
            sentences,
            is_separators,
    ):
        # Каждое предложение разделивает на некоторые последовательности, каждое предсказывает
        # о наличии пробела (1) или отсутствии (0)
        char_sequences_list = []
        space_predict_list = []
        for i in range(len(sentences)):
            s = sentences[i]
            is_sep = is_separators[i]
            for j in range(self.len_chr_lookback):
                s = [BasicPreprocessor.PAD_ID] + s
                is_sep = [self.HIDDEN_STATE_IS_WORD_SEPARATOR] + is_sep
            for j in range(self.len_chr_lookfwd):
                # TODO Instead of 0, put correct variable
                s += [BasicPreprocessor.PAD_ID]
                is_sep += [self.HIDDEN_STATE_IS_WORD_SEPARATOR]
            s_len = len(s)
            #print('For sentence\n\r"' + str(s) + '"\n\r' + str(is_sep))
            for j in range(0, s_len - self.len_chr_lookfwd, 1):
                end_index = j + self.len_chr_lookback + 1 + self.len_chr_lookfwd
                if end_index > s_len:
                    break
                chr_seq = [ s[j:end_index] ]
                sp_predict = [ is_sep[j + self.len_chr_lookback] ]
                char_sequences_list.append(chr_seq)
                space_predict_list.append(sp_predict)
                #print('Char seq: "' + str(chr_seq) + '" sep = ' + str(sp_predict))
            #raise Exception('debiug')
        return char_sequences_list, space_predict_list

    def __preprocess(self):
        # TODO Do proper embedding model, etc. Below only to demo
        self.max_sent_len = BasicPreprocessor.extract_max_length(corpora=self.sentences)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Max len = ' + str(self.max_sent_len)
        )
        self.indexed_dict = BasicPreprocessor.create_indexed_dictionary(
            sentences = self.sentences,
            include_indexed_dict_base = True,
        )
        self.total_words = len(self.indexed_dict)
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Total words = ' + str(self.total_words) + ', Indexed dict: ' + str(self.indexed_dict)
        )
        self.sentences_index = BasicPreprocessor.sentences_to_indexes(
            sentences = self.sentences,
            indexed_dict = self.indexed_dict,
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences numerized: ' + str(self.sentences_index)
        )
        check_sentences = BasicPreprocessor.indexes_to_sentences(
            indexes = self.sentences_index,
            indexed_dict = self.indexed_dict,
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences check: ' + str(check_sentences)
        )
        self.sentences_index_samelen = []
        self.is_separators_samelen = []

        for i in range(len(self.sentences_index)):
            sent = self.sentences_index[i]
            padding_len = self.max_sent_len - len(sent)
            pad_sentence = ([BasicPreprocessor.PAD_ID]*padding_len) + sent
            pad_is_separator = ([1]*padding_len) + self.is_separators_list[i]
            self.sentences_index_samelen.append(pad_sentence)
            self.is_separators_samelen.append(pad_is_separator)

        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences padded: ' + str(self.sentences_index_samelen)
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Is separators padded: ' + str(self.is_separators_samelen)
        )
        check_sentences = BasicPreprocessor.indexes_to_sentences(
            indexes = self.sentences_index_samelen,
            indexed_dict = self.indexed_dict,
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Sentences check: ' + str(check_sentences)
        )
        chars_seq_list, is_separator_list = self.get_sequence_and_prediction_x_y(
            sentences = self.sentences_index_samelen,
            is_separators = self.is_separators_samelen,
        )
        self.X = np.array(chars_seq_list)
        self.Y = np.array(is_separator_list)
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': X: ' + str(self.X)
        )
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Y: ' + str(self.Y)
        )
        return


    """
    Мы создаем простую модель основаная на sequential NN
    Входные данные преобразовывает в векторы слов (модель слов embedding), выходные данные представляет собой 1 или 0,
    Например входное предложение '数学は常に正しい？' будет разбитое по словам к '数学 は 常に 正しい',
    то выходные данные будет [0,1,1,0,1,0,0,1] или более четко [数/0, 学/1, は/1, 常/0, に/1, 正/0, し/0, い/1]
    Но тренинг проходит такого - входные данные состоются векторами длины 5 слов, и выходный только 0 или 1,
    означающий что третым словом является разделитель слов или нет
    """
    def tokenize_model(
            self
    ):
        """
        TODO
           - Следует выделить одно предложение в одну партию (batch), сохраняющая состояние последовательности
             слов в предложении в LSTM
        """
        model = Sequential()

        # Use the text vectorization layer to normalize, split, and map strings to
        # integers. Note that the layer uses the custom standardization defined above.
        # Set maximum_sequence length as all samples are not of the same length.
        # model.add(TextVectorization(
        #     standardize  = custom_standardization,
        #     max_tokens   = self.total_words,
        #     output_mode  = 'int',
        #     output_sequence_length = sequence_length
        # ))

        # Input is max sentence length of dimension (None, max_sent_len)
        # Output is (None, max_sent_len, embedding_vector_len)
        # model.add(Embedding(
        #     # input_dim: Size of the vocabulary, i.e. maximum integer index + 1.
        #     input_dim    = self.total_words,
        #     # output_dim: Dimension of the dense embedding.
        #     output_dim   = self.embedding_vector_len,
        #     # Length of input sequences, when it is constant. This argument is required if you are going to connect
        #     # Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).
        #     input_length = self.max_sent_len
        # ))
        Log.info('Input shape = ' + str(self.X.shape[1:]))
        model.add(LSTM(
            input_shape      = self.X.shape[1:],
            # units            = self.embedding_vector_len,
            units            = 300,
            return_state     = False,
            return_sequences = False,
            activation       = 'tanh',
            # Нет
            stateful         = False,
        ))
        # model.add(Dropout(0.2))
        # model.add(LSTM(
        #     units            = 1,
        #     return_state     = False,
        #     return_sequences = True,
        #     activation       = 'softmax',
        # ))

        # model.add(Dense(
        #     units      = 100,
        #     activation = 'relu',
        #     # kernel_regularizer=regularizers.l2(0.01)
        # ))
        model.add(Dense(
            units      = 1,
            activation = 'linear'
        ))

        # model.compile(optimizer='rmsprop', loss='mse')
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        Log.important(model.summary())

        model.fit(self.X, self.Y, epochs=20, batch_size=100 , verbose=1)

        y = model.predict(x=self.X)
        # Convert to 0, 1
        y = np.round(y, decimals=0)
        # print('For ' + str(self.X) + ', y: '+ str(y) + ', expected: ' + str(self.Y) + ', y shape ' + str(y.shape))
        correct = np.reshape(1*(y == self.Y), newshape=(len(self.Y)))
        correct_pct = np.round(100 * np.sum(correct) / len(correct), decimals=2)
        print('Correct %: ' + str(correct_pct))
        # self.check_result_one_by_one(model_tk=model)
        return

    def check_result_one_by_one(self, model_tk):
        correct = []
        for i in range(len(self.X)):
            x = self.X[i]
            y = model_tk.predict(x=np.array([x]))
            # Convert to characters
            x_words = BasicPreprocessor.indexes_to_sentences(
                indexes      = x,
                indexed_dict = self.indexed_dict,
            )
            expected = int(self.Y[i])*1
            observed = round(y[0][0])
            is_correct = observed == expected
            print('For ' + str(x_words) + ', y: '+ str(y) + ', Expected: ' + str(self.Y[i]) + ', Correct = ' + str(is_correct))
            correct.append(is_correct)

            np_correct = np.array(correct)
            total_len = len(np_correct)
            count_correct = np.sum(np_correct)
            percentage_correct = round(100 * count_correct / total_len, 2)
            print('   Correct = ' + str(np.sum(np_correct)) + ' of ' + str(len(np_correct)) + ', ' + str(percentage_correct) + '%')

    @staticmethod
    def tokenize_jp(sentences_list):
        tokens_tags_list = []
        tokens_list = []
        tags_list = []
        for s in sentences_list:
            words_tags = nagisa.tagging(s)
            tokens_tags_list.append(str(words_tags))
            tokens_list.append(words_tags.words)
            tags_list.append(words_tags.postags)
        return tokens_tags_list, tokens_list, tags_list

    def word_tokenize(self, sentences_list):
        sentences_segmt = [s.split(' ') for s in sentences_list]

        # Remove basic punctuations stuck to word
        sentences_cleanpunc = [BasicPreprocessor.clean_punctuations(sentence=s) for s in sentences_segmt]
        for i in range(len(sentences_cleanpunc)):
            Log.debug(
                #str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno),
                #+ ': Text "' + str(sentences_segmt[i])
                #+ '" clean punctuations to: ' + str(sentences_cleanpunc[i])
                sentences_cleanpunc[i]
            )


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO

    # EXAMPLE_SENT_TRAIN_LIST = [
    #     ('数学は常に正しい？', '数学 は 常に 正しい', [0,1,1,0,1,0,0,1,1]),
    #     ('合理的な人は世界に適応します', '合理 的 な 人 は 世界 に 適応 し ます', [0,1,1,1,1,1,0,1,1,0,1,1,0,1]),
    #     ('進歩は非合理的な人々に依存します', '進歩 は 非 合理 的 な 人々 に 依存 し ます', [0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1]),
    # ]
    # sentences_list = [s[0] for s in EXAMPLE_SENT_TRAIN_LIST]
    # tokenized_list = [s[1].split(' ') for s in EXAMPLE_SENT_TRAIN_LIST]
    # is_sep_list = [s[2] for s in EXAMPLE_SENT_TRAIN_LIST]

    SCRAPE_FROM_WIKI = False

    if SCRAPE_FROM_WIKI:
        # Пример данных из википедии
        sentences_list = WordSegmentationModel.get_training_data_by_scraping_urls(
            url_list = [
                'https://ja.wikipedia.org/wiki/ソニー',
                'https://ja.wikipedia.org/wiki/2020年東京オリンピック',
                'https://ja.wikipedia.org/wiki/新型コロナウイルス感染症の世界的流行_(2019年-)',
                'https://ja.wikipedia.org/wiki/SARSコロナウイルス2',
                'https://ja.wikipedia.org/wiki/新型コロナウイルス感染拡大による東京オリンピック・パラリンピックへの影響',
            ],
            tag_to_find = 'p',
            min_char_per_sent = 10,
            max_char_per_sent = 30,
            write_to_filepath = None,
        )
        print('***** TOTAL SCRAPED = ' + str(len(sentences_list)))
    else:
        f = open(file='sample.japanese.txt', mode='r', encoding='utf-8')
        sentences_list = WordSegmentationModel.get_training_data_from_file(
            filepath = 'sample.japanese.txt',
            min_char_per_sent = 10,
            max_char_per_sent = 30,
        )
        # sentences_list = [s for s in sentences_list if (len(s) >= 10) and (len(s) <= 30)]
        print('***** TOTAL READ = ' + str(len(sentences_list)))
        print(sentences_list)

    tokens_list, is_sep_list = WordSegmentationModel.extract_tokens(
        sentences_list=sentences_list
    )

    t = WordSegmentationModel(
        sentences_list = sentences_list,
        is_separators_list = is_sep_list,
        embedding_vector_len = 2,
    )
    s = 'したがって、進歩は非合理的な人々に依存します。'
    s_t = t.tokenize_jp(sentences_list=[s])
    print(s_t)
    # exit(0)

    # tokens_tags_list, tokens_list, tags_list = t.tokenize_jp(sentences_list=sentences_list)
    # print(tokens_tags_list)
    # print(tokens_list)
    # print(tags_list)

    t.tokenize_model()

