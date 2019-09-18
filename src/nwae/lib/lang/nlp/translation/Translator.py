# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.lib.lang.LangFeatures as lf
import nwae.utils.Profiling as prf


class Translator:

    def __init__(
            self,
            dest_lang,
            nlp_download_dir = None
    ):
        try:
            import nltk
            import googletrans

            self.dest_lang = dest_lang
            nltk.download('punkt', download_dir=nlp_download_dir)
            nltk.data.path.append(nlp_download_dir)

            self.translator = googletrans.Translator()
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Exception instantiating Translator object for destination language "'\
                + str(self.dest_lang) + '": ' + str(ex) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    def detect(
            self,
            sentence
    ):
        start_time = prf.Profiling.start()
        det = self.translator.detect(
                text = sentence
            )
        lg.Log.info(
            'Lang detection of "' + str(sentence) + '" took '
            + str(prf.Profiling.get_time_dif_str(start_time, prf.Profiling.stop()))
        )
        return det.lang

    def translate(
            self,
            sentence
    ):
        try:
            start_time = prf.Profiling.start()
            from nltk import sent_tokenize
            token = sent_tokenize(sentence)

            s = ''
            for tt in token:
                translatedText = self.translator.translate(tt, dest=self.dest_lang)
                s = s + str(translatedText.text)
            lg.Log.info(
                'Lang translation of "' + str(sentence) + '" to "' + str(s) + '" took '
                + str(prf.Profiling.get_time_dif_str(start_time, prf.Profiling.stop()))
            )
            return s
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Exception translating sentence "' + str(sentence) + '": ' + str(ex) + '.'
            lg.Log.error(errmsg)
            raise Exception(errmsg)


if __name__ == '__main__':
    tl = Translator(
        dest_lang = lf.LangFeatures.LANG_ZH_CN
    )

    src = 'Today is a rainy day'
    print(tl.detect(sentence=src))
    s = tl.translate(
        sentence = src
    )
    print(s)
