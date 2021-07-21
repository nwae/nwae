# -*- coding: utf-8 -*-

from nwae.lang.config.Config import Config
from nwae.lang.LangFeatures import LangFeatures
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
from nwae.utils.Log import Log
import nwae.utils.UnitTest as ut


class UtTxtPreprocessor:

    TESTS = {
        LangFeatures.LANG_ZH: [
            #
            # Empty Test
            #
            ['', []],
            #
            # URI Special Symbol Tests
            #
            ['存款10次http://abc.com', ['存款', BasicPreprocessor.W_NUM, '次', BasicPreprocessor.W_URI]],
            # '网址' replaced with '网站' root word
            ['网址 https://wangzhi.com/?a=1&b=2 对吗？', ['网站', BasicPreprocessor.W_URI, '对', '吗', '？']],
            #
            # Number Special Symbol Tests
            #
            ['2019年 12月 26日 俄罗斯部署高超音速武器 取得全球领先',
             [BasicPreprocessor.W_NUM,'年',BasicPreprocessor.W_NUM,'月',BasicPreprocessor.W_NUM,'日','俄罗斯','部署','高超','音速','武器','取得','全球','领先']],
            #
            # Username Special Symbol Tests
            #
            # Complicated username, the '.' will split the word due to word segmentation
            ['用户名 li88jin_99.000__f8', ['用户名', BasicPreprocessor.W_USERNAME_NONWORD, '.', BasicPreprocessor.W_USERNAME_NONWORD]],
            # Characters only is not a username
            ['用户名 notusername', ['用户名','notusername']],
            # Characters with punctuations '.', '-', '_' is a valid username
            ['用户名 is_username', ['用户名', BasicPreprocessor.W_USERNAME_NONWORD]],
            # The '.' will split the usernames due to word segmentation
            ['用户名 is_user.name', ['用户名', BasicPreprocessor.W_USERNAME_NONWORD, '.', 'name']],
            ['用户名 is_user.name-ok.', ['用户名', BasicPreprocessor.W_USERNAME_NONWORD, '.', BasicPreprocessor.W_USERNAME_NONWORD, '.']],
            #
            # No effect to other languages
            #
            ['中文很难english language 한국어 중국 พูดไทย русский язык kiểm tra.',
             ['中文', '很', '难', 'english', 'language', '한국어', '중국', 'พูดไทย', 'русский', 'язык', 'kiểm', 'tra', '.']],
        ],
        LangFeatures.LANG_TH: [
            #
            # URI Special Symbol Tests
            #
            # '网址' replaced with '网站' root word
            ['เว็บไซต์ https://wangzhi.com/?a=1&b=2 ถูก', ['เว็บไซต์', BasicPreprocessor.W_URI, 'ถูก']],
            #
            # Number Special Symbol Tests
            #
            ['ปี2019', ['ปี', BasicPreprocessor.W_NUM]],
            ['ปั่นสล็อต100ครั้ง', ['ปั่น', 'สล็อต', BasicPreprocessor.W_NUM, 'ครั้ง']],
            # The '.' will split the usernames due to word segmentation
            ['อูเสอgeng.mahk_mahk123ได้', ['อูเสอ', 'geng', '.', BasicPreprocessor.W_USERNAME_NONWORD, 'ได้']],
            # Only words should not be treated as username
            ['อูเสอ notusername is_username ได้', ['อูเสอ', 'notusername', BasicPreprocessor.W_USERNAME_NONWORD, 'ได้']],
            ['อยากทำพันธมิตร', ['อยาก', 'ทำ', 'พันธมิตร']],
            #
            # No effect to other languages
            #
            ['中文很难 english language 한국어 중국 พูดไทย русский язык kiểm tra.',
             ['中文很难', 'english', 'language', '한국어', '중국', 'พูด', 'ไทย', 'русский', 'язык', 'kiểm', 'tra', '.']],
        ],
        LangFeatures.LANG_VI: [
            ['đây là bài kiểm tra đơn vị đầu tiên cho tiếng việt', ['đây','là', 'bài', 'kiểm tra', 'đơn vị', 'đầu tiên', 'cho', 'tiếng', 'việt']],
            #
            # No effect to other languages
            #
            ['中文很难 english language 한국어 중국 พูดไทย русский язык kiểm tra.',
             ['中文很难', 'english', 'language', '한국어', '중국', 'พูดไทย', 'русский', 'язык', 'kiểm tra', '.']],
        ]
    }

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def __init_txt_preprocessor(self, lang):
        self.lang = lang
        self.txt_preprocessor = TxtPreprocessor(
            identifier_string      = 'unit test ' + str(self.lang),
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = self.lang,
            dirpath_synonymlist    = self.ut_params.dirpath_synonymlist,
            postfix_synonymlist    = self.ut_params.postfix_synonymlist,
            dir_wordlist           = self.ut_params.dirpath_wordlist,
            postfix_wordlist       = self.ut_params.postfix_wordlist,
            dir_wordlist_app       = self.ut_params.dirpath_app_wordlist,
            postfix_wordlist_app   = self.ut_params.postfix_app_wordlist,
            do_spelling_correction = False,
            do_word_stemming       = True,
            do_profiling           = False
        )

    def __run_lang_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        for txt_expected in UtTxtPreprocessor.TESTS[self.lang]:
            txt = txt_expected[0]
            expected = txt_expected[1]
            observed = self.txt_preprocessor.process_text(
                inputtext = txt,
                return_as_string = False,
                use_special_symbol_username_nonword = True
            )
            res_final.update_bool(res_bool = ut.UnitTest.assert_true(
                observed = observed,
                expected = expected,
                test_comment = 'test "' + str(txt) + '"'
            ))

        Log.important(
            '***** ' + str(self.lang) + ' PASSED ' + str(res_final.count_ok)
            + ', FAILED ' + str(res_final.count_fail) + ' *****'
        )
        return res_final

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)
        for lang in [LangFeatures.LANG_ZH, LangFeatures.LANG_TH, LangFeatures.LANG_VI]:
            self.__init_txt_preprocessor(lang=lang)
            res = self.__run_lang_unit_test()
            res_final.update(other_res_obj=res)

        return res_final

if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None
    )
    print('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_WARNING
    UtTxtPreprocessor(ut_params=ut_params).run_unit_test()

