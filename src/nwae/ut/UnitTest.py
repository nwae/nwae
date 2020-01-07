# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from nwae.utils.UnitTest import ResultObj, UnitTestParams
import nwae.config.Config as cf
from nwae.utils.ObjectPersistence import UnitTestObjectPersistence
from mex.UnitTest import UnitTestMex
from nwae.lib.lang.nlp.ut.UtWordSegmentation import UnitTestWordSegmentation
from nwae.lib.lang.preprocessing.ut.UtTxtPreprocessor import UtTxtPreprocessor
from nwae.lib.lang.preprocessing.ut.UtTrDataPreprocessor import UtTrDataPreprocessor
from nwae.lib.math.ml.metricspace.ut.UtMetricSpaceModel import UnitTestMetricSpaceModel
import nwae.lib.math.ml.ModelHelper as modelHelper


#
# We run all the available unit tests from all modules here
# PYTHONPATH=".:/usr/local/git/nwae/nwae.utils/src:/usr/local/git/nwae/mex/src" /usr/local/bin/python3.6 nwae/ut/UnitTest.py
#
class NwaeUnitTest:

    def __init__(self, ut_params):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = UnitTestParams()
        return

    def run_unit_tests(self):
        all_pass = 0
        all_fail = 0

        res = UnitTestObjectPersistence(ut_params=None).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Object Persistence Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UnitTestMex(config=None).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Mex Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
        #
        # Word Tokenization or Segmentation
        #
        res = UnitTestWordSegmentation(ut_params=self.ut_params).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Word Segmentation Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UtTxtPreprocessor(ut_params=self.ut_params).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Text Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UtTrDataPreprocessor(ut_params=self.ut_params).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Training Data Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UnitTestMetricSpaceModel(
            ut_params = self.ut_params,
            identifier_string = 'demo_ut1',
            model_name = modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE
        ).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('MetricSpaceModel Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical('TOTAL PASS = ' + str(all_pass) + ', TOTAL FAIL = ' + str(all_fail))
        return ResultObj(count_ok=all_pass, count_fail=all_fail)


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    )

    ut_params = UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    print('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_ERROR

    NwaeUnitTest(ut_params=ut_params).run_unit_tests()
