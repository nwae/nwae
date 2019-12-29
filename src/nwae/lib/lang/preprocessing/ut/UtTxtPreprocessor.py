# -*- coding: utf-8 -*-

from nwae.config.Config import Config
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lib.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
from nwae.utils.Log import Log


class UtTxtPreprocessor:

    TESTS = {
        LangFeatures.LANG_CN: [
            ['', []],
            #
            # Number Tests
            #
            ['2019年 12月 26日 俄罗斯部署高超音速武器 取得全球领先',
             [BasicPreprocessor.W_NUM,'年',BasicPreprocessor.W_NUM,'月',BasicPreprocessor.W_NUM,'日','俄罗斯','部署','高超','音速','武器','取得','全球','领先']],
            #
            # Username Tests
            #
            # Complicated username
            ['用户名 li88jin_99.000__f8', ['用户名', BasicPreprocessor.W_USERNAME_CHARNUM]],
            # Characters only is not a username
            ['用户名 notusername', ['用户名','notusername']],
            # Characters with punctuations '.', '-', '_' is a valid username
            ['用户名 is_username', ['用户名', BasicPreprocessor.W_USERNAME_CHARNUM]],
            ['用户名 is_user.name', ['用户名', BasicPreprocessor.W_USERNAME_CHARNUM]],
            ['用户名 is_user.name-ok.', ['用户名', BasicPreprocessor.W_USERNAME_CHARNUM, '.']],
        ],
        LangFeatures.LANG_TH: [
            ['ปั่นสล็อต100ครั้ง', ['ปั่น', 'สล็อต', BasicPreprocessor.W_NUM, 'ครั้ง']],
            ['อูเสอgeng.mahk_mahk123ได้', ['อูเสอ', BasicPreprocessor.W_USERNAME_CHARNUM, 'ได้']],
            # Only words should not be treated as username
            ['อูเสอ notusername is_username ได้', ['อูเสอ', 'notusername', BasicPreprocessor.W_USERNAME_CHARNUM, 'ได้']],
            ['อยากทำพันธมิตร', ['อยาก', 'ทำ', 'พันธมิตร']]
        ]}

    def __init__(
            self,
            lang,
            config
    ):
        self.lang = lang
        self.config = config

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
        count_ok = 0
        count_fail = 0
        for txt_expected in UtTxtPreprocessor.TESTS[self.lang]:
            txt = txt_expected[0]
            expected = txt_expected[1]
            res = self.txt_preprocessor.process_text(inputtext=txt)
            if res != expected:
                count_fail += 1
                print('FAIL. Error txt "' + str(txt) + '", got ' + str(res) + ', expected ' + str(expected))
            else:
                count_ok += 1
                print('OK "' + str(txt) + '", result' + str(res))

        print('***** PASSED ' + str(count_ok) + ', FAILED ' + str(count_fail) + ' *****')


if __name__ == '__main__':
    config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class=Config,
        default_config_file=config_file
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_IMPORTANT
    UtTxtPreprocessor(lang = LangFeatures.LANG_CN, config=config).run_ut()
    UtTxtPreprocessor(lang = LangFeatures.LANG_TH, config=config).run_ut()

