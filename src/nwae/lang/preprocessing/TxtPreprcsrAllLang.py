# -*- coding: utf-8 -*-
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lang.LangFeatures import LangFeatures
from nwae.utils.Profiling import Profiling, ProfilingHelper
import nwae.utils.UnitTest as ut
from nwae.lang.detect.LangDetect import LangDetect
import pandas as pd
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
import collections


#
# The very first step in NLP, preprocess text into clean/remodeled text,
# before the embedding model where all text converted to numbers/vectors/tensors for NN input
# All text pre-processing regardless of language should use this class only for consistency
#
class TextPreprscrAllLang:

    NEW_DATA = None

    # class 'list'
    EXAMPLE_TEXTS = [
        'url example text https://www.rome2rio.com/, и ише ',
        # fr
        'Que faire so vous voudriez Étudier en France',
        'Acheter une voiture à dubai;Comment je peux acheter une voiture a Dubai?;',
        # en
        'Olympics organizers are banning all spectators from the games this year after Japan declared a state of '
        'emergency that’s meant to curb a wave of new Covid-19 infections.',
        'Organizers had already banned international spectators from attending and set a cap on domestic crowds '
        'at 50% of capacity, or up to 10,000 people.'
        'The Olympic Charter expressly bans “discrimination of any kind,” as a “Fundamental Principle of Olympism.” ',
        'Japanese sport has a history of corporal punishment against children, known in Japanese as taibatsu. ',
        # zh
        '去年因疫情未如期举行的香港书展今年将于周三（7月14日）在湾仔会展中心开幕，这是香港《国安法》实施后的首个书展。',
        '本次香港书展将一连举行7日。香港出版界人士对BBC中文表示，本次书展已经有“白色恐怖”气氛，许多参展商因为摸不清《国安法》红线只能进行自我审查。',
        # ru
        'Чат-бот Lee Ludа, разработанный сеульским стартапом Scatter Lab и использовавший личность 20-летней '
        'студентки университета, был удалён из Facebook messenger на этой неделе',
        'По словам издания, это уже не первый случай, когда искусственный интеллект сталкивается с обвинениями '
        'в нетерпимости и дискриминации.',
        'Аккаунт популярного южнокорейского чат-бота был заблокирован в Facebook после жалоб на ненавистнические '
        'высказывания в адрес сексуальных меньшинств.',
        'Как передаёт газета, в одном из диалогов с пользователями бот по имени Lee Ludа назвала лесбиянок '
        '«жуткими» и призналась, что ненавидит их.',
        'Чат-боты в банковской сфере, телекоме и онлайн-ретейле решают две трети всех клиентских запросов, пишет '
        '«Коммерсантъ» со ссылкой на исследование Markswebb, посвященное эффективности текстовых роботов.',
    ]

    def __init__(
            self,
            dir_wordlist,
            postfix_wordlist,
            dir_app_wordlist,
            postfix_app_wordlist,
            dir_synlist,
            postfix_synlist,
            stopwords_list = (),
    ):
        self.dir_wordlist = dir_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dir_app_wordlist = dir_app_wordlist
        self.postfix_app_wordlist = postfix_app_wordlist
        self.dir_synlist = dir_synlist
        self.postfix_synlist = postfix_synlist
        self.stopwords_list = stopwords_list

        self.lang_detect = LangDetect()
        # By language, created when need
        self.word_segmenter = {}
        self.txt_preprcsr_by_lang = {}
        self.profiler = ProfilingHelper(profiler_name=str(self.__class__))
        return

    """
    Take a list of sentences, output another list of cleaned sentences
    Цель заключается в том, чтобы дальнейшая обработка будут удобнее, вроде написания польных слов
    с гласными в арабском или без Канджи в японском для детей
    """
    def preprocess_list_all_langs(
            self,
            sentences_list,
            # The output required may differ for different further processings
            # Some may require POS Tagging, some may require lemmatization, some may require to remove
            # stop words, etc
            algorithm=None,
    ):
        langs_list = self.detect_lang(sentences_list = sentences_list, method='nwae')
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Done detecting ' + str(len(sentences_list)) + ' sentence languages: ' + str(langs_list)
        )
        # Get most common lang
        langs_counter = collections.Counter(langs_list).most_common()
        self.lang_default = langs_counter[0][0]
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Most common language detected "' + str(self.lang_default) + '" from ' + str(langs_counter)
        )
        unique_langs = [l for l in list(set(langs_list)) if l]
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Unique langs found: ' + str(unique_langs)
        )
        for lang in unique_langs:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Creating lang "' + str(lang) + '" word segmenter.'
            )
            self.txt_preprcsr_by_lang[lang] = TxtPreprocessor(
                identifier_string    = lang,
                dir_path_model       = None,
                model_features_list  = None,
                lang                 = lang,
                dir_wordlist         = self.dir_wordlist,
                postfix_wordlist     = self.postfix_wordlist,
                dir_wordlist_app     = self.dir_app_wordlist,
                postfix_wordlist_app = self.postfix_app_wordlist,
                dirpath_synonymlist  = self.dir_synlist,
                postfix_synonymlist  = self.postfix_synlist,
                stopwords_list       = self.stopwords_list,
            )
        sentences_list_processed = []
        for i in range(len(sentences_list)):
            sent = sentences_list[i]
            lang = langs_list[i]
            if not lang:
                lang = self.lang_default
            sent_processed = self.txt_preprcsr_by_lang[lang].process_text(
                inputtext = sent,
            )
            # commented out this part for now
            Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Preprocessed sentence "' + str(sent) + '" to "' + str(sent_processed) + '"'
            )
            sentences_list_processed.append(sent_processed)

        return sentences_list_processed

    def detect_lang(
            self,
            sentences_list,
            method = 'nwae',
    ):
        lang_list = []
        start_time = Profiling.start()
        for s in sentences_list:
            lang = self.lang_detect.detect(text=s)
            # Get top detected language only
            # If returned is non-empty list
            if lang:
                lang = lang[0]
            else:
                lang = ''
            Log.debug(
                'Lang "' + str(lang) + '" detected for "' + str(s) + '"'
            )
            lang_list.append(lang)
        self.profiler.profile_time(start_time=start_time, additional_info=method)
        return lang_list


class TextPreprscrAllLangUnitTest:
    def __init__(self):
        return

    def get_stats_lang_detect(
            self,
            sentences_list,
            langs_real,
            langs_detected,
    ):
        correct_count = 0
        total_count = len(langs_real)
        for i in range(total_count):
            lang_det = langs_detected[i]
            lang_real = langs_real[i]
            correct_result = lang_real == lang_det
            if not correct_result:
                Log.debug(
                    'Detected "' + str(lang_det) + '" for supposed "' + str(lang_real)
                    + '" sent "' + str(sentences_list[i]) + '"'
                )
            correct_count += 1*(correct_result)
        correct_pct = round(100 * correct_count / total_count, 2)
        return correct_pct, correct_count, total_count

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        from nwae.lang.config.Config import Config
        config = Config.get_cmdline_params_and_init_config_singleton(
            Derived_Class       = Config,
            default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
        )

        # DATAFRAME = pd.read_csv('task1.3.csv', sep=";")
        df = pd.read_csv('/usr/local/git/nwae/nwae.lang/data/sample.intents.csv', sep=";")
        print(df)
        # from class 'pandas.core.series.Series' to class 'list' (here are all sents, same as in EXAMPLE_TEXTS):
        sents = pd.Series(df['sentence'], dtype="string").tolist()
        langs = pd.Series(df['lang'], dtype="string").tolist()
        print('Langs ' + str(langs))

        tp = TextPreprscrAllLang(
            dir_wordlist         = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
            dir_app_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
            postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
            dir_synlist          = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synlist      = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        )

        tp.preprocess_list_all_langs(
            sentences_list = sents
        )

        langs_detected = tp.detect_lang(sentences_list=sents)
        correct_pct, correct_count, total_count = self.get_stats_lang_detect(
            sentences_list = sents,
            langs_real     = langs,
            langs_detected = langs_detected,
        )
        Log.important(
            'Method language detection. Correct ' + str(correct_pct) + '%, ' + str(correct_count) + '/' + str(total_count)
        )

        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    TextPreprscrAllLangUnitTest().run_unit_test()
    exit(0)
