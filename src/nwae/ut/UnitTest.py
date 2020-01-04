# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
import nwae.config.Config as cf
from mex.UnitTest import UnitTestMex
from nwae.lib.lang.nlp.ut.UtWordSegmentation import UnitTestWordSegmentation
from nwae.lib.lang.preprocessing.ut.UtTxtPreprocessor import UtTxtPreprocessor
from nwae.lib.math.ml.metricspace.ut.UtMetricSpaceModel import UnitTestMetricSpaceModel
import nwae.lib.math.ml.ModelHelper as modelHelper


#
# We run all the available unit tests from all modules here
# PYTHONPATH=".:/usr/local/git/nwae/nwae.utils/src:/usr/local/git/nwae/mex/src" /usr/local/bin/python3.8 nwae/ut/UnitTest.py
#
class NwaeUnitTest:

    def __init__(self, config):
        self.config = config
        return

    def run_unit_tests(self):
        all_pass = 0
        all_fail = 0

        res = UnitTestMex(config=None).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Mex Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
        #
        # Word Tokenization or Segmentation
        #
        res = UnitTestWordSegmentation(config=self.config).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Word Segmentation Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UtTxtPreprocessor(config=self.config).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Text Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UnitTestMetricSpaceModel(
            config = self.config,
            identifier_string = 'demo_ut1',
            model_name = modelHelper.ModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE
        ).run_unit_test()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('MetricSpaceModel Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical('TOTAL PASS = ' + str(all_pass) + ', TOTAL FAIL = ' + str(all_fail))
        return


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_ERROR

    NwaeUnitTest(config=config).run_unit_tests()
