# -*- coding: utf-8 -*-

from nwae.config.Config import Config
from nwae.lib.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
from nwae.utils.Log import Log


class UtTxtPreprocessor:

    def __init__(
            self,
            lang,
            config_file = None
    ):
        if config_file is None:
            config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
        self.config = Config.get_cmdline_params_and_init_config_singleton(
            Derived_Class       = Config,
            default_config_file = config_file
        )
        self.lang = lang

        self.txt_preprocessor = TxtPreprocessor(
            identifier_string      = 'unit test ' + str(self.lang),
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = self.lang,
            dirpath_synonymlist    = self.config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synonymlist    = self.config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
            dir_wordlist           = self.config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist       = self.config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
            dir_wordlist_app       = self.config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
            postfix_wordlist_app   = self.config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
            do_spelling_correction = False,
            do_word_stemming       = False,
            do_profiling           = False
        )

    def run_ut(self):
        texts = [
            'ปั่นสล็อต100ครั้ง',
            'อูเสอgeng.mahk_mahk123ได้'
        ]

        for txt in texts:
            self.txt_preprocessor.process_text(inputtext=txt)


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO
    UtTxtPreprocessor(lang = 'th').run_ut()

