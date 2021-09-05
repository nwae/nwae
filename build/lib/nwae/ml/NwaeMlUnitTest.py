# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.utils.UnitTest as uthelper
import nwae.ml.config.Config as cf
from nwae.utils.ObjectPersistence import UnitTestObjectPersistence
from nwae.lang.NwaeLangUnitTest import NwaeLangUnitTest
from nwae.math.NwaeMathUnitTest import NwaeMathUnitTest
from nwae.ml.text.preprocessing.TrDataPreprocessorUnitTest import TrDataPreprocessorUnitTest
from nwae.ml.TrainingDataModel import TrainingDataModelUnitTest
from nwae.ml.metricspace.ut.UtMetricSpaceModel import UnitTestMetricSpaceModel
from nwae.ml.PredictClass import PredictClassUnitTest
from nwae.ml.modelhelper.TextModelHelper import TextModelHelper


#
# We run all the available unit tests from all modules here
# PYTHONPATH=".:/usr/local/git/nwae/nwae.utils/src:/usr/local/git/nwae/mex/src" /usr/local/bin/python3.6 nwae/ut/UnitTest.py
#
class NwaeMlUnitTest:

    def __init__(self, ut_params):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = uthelper.UnitTestParams()
        return

    def run_unit_tests(self):
        res_final = uthelper.ResultObj(count_ok=0, count_fail=0)

        res = UnitTestObjectPersistence(ut_params=None).run_unit_test()
        if res.count_fail > 0: raise Exception('Object Persistence failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.utils>> Object Persistence Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = NwaeLangUnitTest(ut_params=self.ut_params).run_unit_tests()
        if res.count_fail > 0: raise Exception('nwae.lang failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        # Log.critical('Project <<nwae.lang>> Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = NwaeMathUnitTest(ut_params=None).run_unit_tests()
        if res.count_fail > 0: raise Exception('nwae.math failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        # Log.critical('Project <<nwae.math>>  Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = TrDataPreprocessorUnitTest(ut_params=self.ut_params).run_unit_test()
        if res.count_fail > 0: raise Exception('TD Data Preprocessor failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.ml>> TD Data Preprocessor Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = TrainingDataModelUnitTest(ut_params=self.ut_params).run_unit_test()
        if res.count_fail > 0: raise Exception('Training Data Model failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.ml>> Training Data Model Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = UnitTestMetricSpaceModel(
            ut_params  = self.ut_params,
            model_name = TextModelHelper.MODEL_NAME_HYPERSPHERE_METRICSPACE
        ).run_unit_test()
        if res.count_fail > 0: raise Exception('MetricSpaceModel failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.ml>> MetricSpaceModel Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = PredictClassUnitTest(ut_params=self.ut_params).run_unit_test()
        if res.count_fail > 0: raise Exception('PredictClass failed: ' + str(res.count_fail))
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.ml>> PredictClass Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        # try:
        #     # Try to import some Keras module to see if available
        #     from keras.utils import to_categorical
        #     test_nn_dense = True
        # except Exception as ex_load:
        #     errmsg = str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
        #              + ': Could not test NN Dense Model: ' + str(ex_load)
        #     Log.error(errmsg)
        #     test_nn_dense = False
        # if test_nn_dense:
        #     from nwae.ml.nndense.NnDenseModelUnitTest import NnDenseModelUnitTest
        #     res = NnDenseModelUnitTest(
        #         ut_params = self.ut_params
        #     ).run_unit_test()
        #     if res.count_fail > 0: raise Exception('NN Dense Model failed: ' + str(res.count_fail))
        #     res_final.update(other_res_obj=res)
        #     Log.critical('<<nwae.ml>> NN Dense Model Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': PROJECT <<nwae.ml>> TOTAL PASS = ' + str(res_final.count_ok)
            + ', TOTAL FAIL = ' + str(res_final.count_fail)
        )
        return res_final


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_DEFAULT
    )

    ut_params = uthelper.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_ERROR

    res = NwaeMlUnitTest(ut_params=ut_params).run_unit_tests()
    exit(res.count_fail)
