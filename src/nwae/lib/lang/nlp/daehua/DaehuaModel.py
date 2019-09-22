# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.config.Config as cf
import nwae.lib.lang.LangFeatures  as lf
import json


class DaehuaModel:

    # No specific value type
    PARAM_GENERAL = 'param_general'
    # Extract date type which may be different for different
    PARAM_DATE = 'param_date'
    PARAM_NAME = 'param_name'

    def __init__(
            self,
            lang,
            config
    ):
        self.lang = lang
        self.config = config

        self.daehua_dir = self.config.get_config(param=cf.Config.PARAM_NLP_DAEHUA_DIR)
        self.daehua_pattern_filepath = self.config.get_config(param=cf.Config.PARAM_NLP_DAEHUA_PATTERN_JSON_FILE)

        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Using daehua "' + str(self.daehua_dir)
            + '", and pattern file "' + str(self.daehua_pattern_filepath)
        )
        self.__load_patterns()
        return

    def __load_patterns(self):
        try:
            with open(self.daehua_pattern_filepath, encoding='utf-8') as h:
                d = json.load(h)
                print(d)
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Error loading patterns from file "' + str(self.daehua_pattern_filepath)\
                + '", got exception message: ' + str(ex)
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    def get_param_value_from_sentence(
            self,
            # Sentence in word array. e.g. ['내','생일은','9','월','26','']
            sentence_word_array,
            param
    ):
        val = None
        return val


if __name__ == '__main__':
    cf_obj = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config
    )
    cm_obj = DaehuaModel(
        lang = lf.LangFeatures.LANG_CN,
        config = cf_obj
    )
