# -*- coding: utf-8 -*-

# TODO Replace all nltk with in-house
import nltk
import nltk.stem.wordnet as wordnet
import nltk.stem.porter as porter
import nltk.stem.snowball as snowball
import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.utils.Profiling as prf
import nwae.lang.LangFeatures as lf
from nwae.utils.networking.Ssl import Ssl
from nwae.lang.nlp.lemma.LemmatizerKorean import LemmatizerKorean


class Lemmatizer:

    TYPE_PORTER_STEMMER = 'porter-stemmer'
    TYPE_SNOWBALL_STEMMER = 'snowball-stemmer'
    TYPE_WORDNET_LEMMATIZER = 'wordnet-lemmatizer'

    SUPPORTED_LANGUAGES = [
        lf.LangFeatures.LANG_EN,
        lf.LangFeatures.LANG_KO,
        # TODO Below
        lf.LangFeatures.LANG_RU,
        # lf.LangFeatures.LANG_FR,
        # lf.LangFeatures.LANG_ES,
    ]

    def __init__(
            self,
            lang = lf.LangFeatures.LANG_EN,
            # Choice of stemmer type only applies to english
            stemmer_type = TYPE_PORTER_STEMMER
    ):
        self.lang = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        self.stemmer_type = stemmer_type

        # 바보 nltk is broken, https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
        # TODO Write our own Lemmatizer
        Ssl.disable_ssl_check()

        if lang not in Lemmatizer.SUPPORTED_LANGUAGES:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Stemmer for language "' + str(lang) + '" not supported.'
            lg.Log.warning(errmsg)
            raise Exception(errmsg)
        else:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Stemmer for lang "' + str(lang) + '" ok'
            )

        self.stemmer = None

        if self.lang == lf.LangFeatures.LANG_EN:
            if self.stemmer_type == Lemmatizer.TYPE_WORDNET_LEMMATIZER:
                nltk.download('wordnet')
                self.stemmer = wordnet.WordNetLemmatizer()
            elif self.stemmer_type == Lemmatizer.TYPE_PORTER_STEMMER:
                self.stemmer = porter.PorterStemmer()
            elif self.stemmer_type == Lemmatizer.TYPE_SNOWBALL_STEMMER:
                self.stemmer = snowball.SnowballStemmer(
                    language = 'english'
                )
            else:
                raise Exception(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ':Unrecognized stemmer type "' + str(self.stemmer_type) + '".'
                )
            # Call once, because only the first one is slow
            self.stem(word='initialize')
        elif self.lang == lf.LangFeatures.LANG_KO:
            self.stemmer = LemmatizerKorean()
        elif self.lang == lf.LangFeatures.LANG_RU:
            self.stemmer = self.stemmer = snowball.SnowballStemmer(language = 'russian')
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Unsupported language "' + str(self.lang) + '"'
            )

        return

    def stem(
            self,
            word
    ):
        try:
            if self.stemmer_type == Lemmatizer.TYPE_WORDNET_LEMMATIZER:
                word_stem = self.stemmer.lemmatize(word = word)
            else:
                word_stem = self.stemmer.stem(word = word)
            lg.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" word "' + str(word) + '", word stem "' + str(word_stem) + '"'
            )
            return word_stem
        except Exception as ex:
            lg.Log.error(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Stem error for lang "' + str(self.lang) + '", word "' + str(word) + '": ' + str(ex)
            )
            return word


if __name__ == '__main__':
    l_lemma = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_WORDNET_LEMMATIZER
    )
    l_porter = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_PORTER_STEMMER
    )
    l_snowball = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_SNOWBALL_STEMMER
    )

    words = [
        'initialize', 'article', 'leaves', 'is', 'are', 'programming', 'programmer',
        'books', 'downloading', 'downloader', 'eating', 'ate', 'beauty',
        'люблю', 'ем',
        '미친', '나가'
    ]

    print('Word --> Wordnet Lemmatizer, Porter Stemmer, Snowball Stemmer')
    a = prf.Profiling.start()
    for w in words:
        print(
            str(w) + ' --> '
            + str(l_lemma.stem(word=w)) + ' (wn)'
            + ', ' + str(l_porter.stem(word=w)) + ' (pt)'
            + ', ' + str(l_snowball.stem(word=w)) + ' (sb)'
        )

    b = prf.Profiling.stop()
    total_time_secs = prf.Profiling.get_time_dif(start=a, stop=b)
    total_time_ms = total_time_secs * 1000
    rps = round(len(words) / total_time_secs, 4)
    tpr = round(1/rps, 4)
    print('Time start: ' + str(a) + ', time end: ' + str(b))
    print('RPS: ' + str(rps) + 'rps, time per request = ' + str(tpr) + 'ms')

    exit(0)
