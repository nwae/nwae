# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
import nwae.utils.UnitTest as uthelper
import nwae.lang.config.Config as cf
from mex.MexUnitTest import UnitTestMex
from nwae.lang.LangFeatures import LangFeaturesUnitTest
from nwae.lang.characters.LangCharacters import LangCharactersUnitTest
from nwae.lang.detect.LangDetectUnitTest import LangDetectUnitTest
from nwae.lang.nlp.WordList import WordlistUnitTest
from nwae.lang.nlp.ut.UtWordSegmentation import UnitTestWordSegmentation
from nwae.lang.preprocessing.BasicPreprocessor import BasicPreprocessorUnitTest
from nwae.lang.preprocessing.ut.UtTxtPreprocessor import UtTxtPreprocessor
from nwae.lang.nlp.sajun.EditDistance import EditDistanceUnitTest
from nwae.lang.nlp.sajun.TrieNode import TrieNodeUnitTest
from nwae.lang.nlp.sajun.SpellCheckWord import SpellCheckWordUnitTest
from nwae.lang.nlp.sajun.SpellCheckSentence import SpellCheckSentenceUnitTest
from nwae.lang.model.FeatureVect import FeatureVectorUnitTest
from nwae.lang.model.WordFreqDocMatrix import WordFreqDocMatrixUnitTest
from nwae.lang.classification.TxtClusterUnitTest import TxtClusterUnitTest


#
# We run all the available unit tests from all modules here
#
class NwaeLangUnitTest:

    def __init__(self, ut_params):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = uthelper.UnitTestParams()
        return

    def run_unit_tests(self):
        res_final = uthelper.ResultObj(count_ok=0, count_fail=0)

        res = UnitTestMex(config=None).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Mex Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = LangFeaturesUnitTest(ut_params=None).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Language Features Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = LangCharactersUnitTest(ut_params=None).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Language Characters Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = LangDetectUnitTest(ut_params=None).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Language Detect Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = WordlistUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Wordlist Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UnitTestWordSegmentation(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Tokenizer Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = BasicPreprocessorUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Basic Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UtTxtPreprocessor(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = EditDistanceUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Edit Distance (DLev, Lev) Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = TrieNodeUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> TrieNode (Edit Distance) Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = SpellCheckWordUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Spell Check Word Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = SpellCheckSentenceUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Spell Check Sentence Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = FeatureVectorUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Feature Vector Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = WordFreqDocMatrixUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Word Freq Doc Matrix Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = TxtClusterUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.lang>> Text Cluster Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical('PROJECT <<nwae.lang>> TOTAL PASS = ' + str(res_final.count_ok) + ', TOTAL FAIL = ' + str(res_final.count_fail))
        return res_final


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae.lang/app.data/config/default.cf'
    )

    ut_params = uthelper.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None,
    )
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_ERROR

    res = NwaeLangUnitTest(ut_params=ut_params).run_unit_tests()
    exit(res.count_fail)
