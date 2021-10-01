# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
import nwae.utils.UnitTest as uthelper
import nwae.math.config.Config as cf
from nwae.utils.sec.Encrypt import EncryptUnitTest
from nwae.math.NumpyUtil import NumpyUtilUnittest
from nwae.math.Cluster import ClusterUnitTest
from nwae.utils.Hash import HashUnitTest
from nwae.math.Obfuscate import ObfuscateUnitTest
from nwae.math.data.Rank import RankUnitTest
from nwae.math.data.Grouping import GroupingUnitTest
from nwae.math.measures.CenterOfMass import CenterOfMassUnitTest
from nwae.math.tree.MultiTree import MultiTreeUnitTest
from nwae.math.fit.markov.HiddenMarkov import HiddenMarkovUnitTest
from nwae.math.recommendation.RecommendDataProfile import RecommendDataProfileUnitTest
from nwae.math.recommendation.RecommendMetric import RecommendMetricUnitTest


#
# We run all the available unit tests from all modules here
# PYTHONPATH=".:/usr/local/git/nwae/nwae.utils/src:/usr/local/git/nwae/mex/src" /usr/local/bin/python3.6 nwae/ut/UnitTest.py
#
class NwaeMathUnitTest:

    def __init__(self, ut_params):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = uthelper.UnitTestParams()
        return

    def run_unit_tests(self):
        res_final = uthelper.ResultObj(count_ok=0, count_fail=0)

        res = EncryptUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.utils>> Encrypt Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = NumpyUtilUnittest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Numpy Util Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = ClusterUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Cluster Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = HashUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.utils>> Hash Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = ObfuscateUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Obfuscate Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = RankUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Data Rank Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = GroupingUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Data Grouping Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = CenterOfMassUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Center of Mass Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = MultiTreeUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Multi Tree Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = HiddenMarkovUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Hidden Markov Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = RecommendDataProfileUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Recommend Data Profile Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        res = RecommendMetricUnitTest(ut_params=self.ut_params).run_unit_test()
        res_final.update(other_res_obj=res)
        Log.critical('<<nwae.math>> Recommend Metric Unit Test PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical('PROJECT <<nwae.math>> TOTAL PASS = ' + str(res_final.count_ok) + ', TOTAL FAIL = ' + str(res_final.count_fail))
        return res_final


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_PATH_DEFAULT
    )

    ut_params = uthelper.UnitTestParams()
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    Log.LOGLEVEL = Log.LOG_LEVEL_ERROR

    res = NwaeMathUnitTest(ut_params=ut_params).run_unit_tests()
    exit(res.count_fail)
