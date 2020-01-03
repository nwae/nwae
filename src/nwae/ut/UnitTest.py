# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
import nwae.config.Config as cf
from nwae.lib.lang.nlp.ut.UtWordSegmentation import UnitTestWordSegmentation


#
# We run all the available unit tests from all modules here
#
class NwaeUnitTest:

    def __init__(self, config):
        self.config = config
        return

    def run_unit_tests(self):
        all_pass = 0
        all_fail = 0
        #
        # Word Tokenization or Segmentation
        #
        res = UnitTestWordSegmentation(config=self.config).run_unit_tests()
        all_pass += res.count_ok
        all_fail += res.count_fail
        Log.critical('Word Segmentation Unit Tests PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))

        Log.critical('TOTAL PASS = ' + str(all_pass) + ', TOTAL FAIL = ' + str(all_fail))
        return


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_WARNING

    NwaeUnitTest(config=config).run_unit_tests()
