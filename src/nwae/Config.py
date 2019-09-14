# -*- coding: utf-8 -*-

import os
import nwae.utils.BaseConfig as baseconfig
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# We put all our common file paths here
#
class Config(baseconfig.BaseConfig):


    PARAM_TOPDIR = 'topdir'
    DEFAULT_VALUE_TOPDIR = None

    PARAM_LOG_LEVEL = 'loglevel'
    DEFAULT_VALUE_LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    PARAM_MODEL_DIR = 'model_dir'
    DEFAULT_VALUE_MODEL_DIR = None

    PARAM_MODEL_IDENTIFIER = 'model_identifier'
    DEFAULT_VALUE_MODEL_IDENTIFIER = None

    #
    # NLP Settings
    #
    PARAM_NLP_DIR_WORDLIST = 'dir_wordlist'
    DEFAULT_VALUE_DIR_WORDLIST = None

    PARAM_NLP_POSTFIX_WORDLIST = 'postfix_wordlist'
    DEFAULT_VALUE_POSTFIX_WORDLIST = '-wordlist.txt'

    PARAM_NLP_POSTFIX_STOPWORDS = 'postfix_stopwords'
    DEFAULT_VALUE_POSTFIX_STOPWORDS = '-stopwords.txt'

    PARAM_NLP_DIR_APP_WORDLIST = 'dir_app_wordlist'
    DEFAULT_VALUE_DIR_APP_WORDLIST = None

    PARAM_NLP_POSTFIX_APP_WORDLIST = 'postfix_app_wordlist'
    DEFAULT_VALUE_POSTFIX_APP_WORDLIST = '.wordlist.app.txt'

    PARAM_NLP_POSTFIX_APP_STOPWORDS = 'postfix_app_stopwords'
    DEFAULT_VALUE_POSTFIX_APP_STOPWORDS = '.stopwords.app.txt'

    PARAM_NLP_DIR_SYNONYMLIST = 'dir_synonymlist'
    DEFAULT_VALUE_DIR_SYNONYMLIST = None

    PARAM_NLP_POSTFIX_SYNONYMLIST = 'postfix_synonymlist'
    DEFAULT_VALUE_POSTFIX_SYNONYMLIST = '.synonymlist.txt'

    PARAM_DEBUG = 'debug'
    DEFAULT_VALUE_DEBUG = False

    PARAM_DO_PROFILING = 'do_profiling'
    DEFAULT_VALUE_DO_PROFILING = False

    def __init__(
            self,
            config_file
    ):
        super(Config, self).__init__(
            config_file = config_file
        )

        try:
            self.reset_default_config()

            lg.Log.LOGLEVEL = float(self.param_value[Config.PARAM_LOG_LEVEL])
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': CONFIG "' + str(Config.PARAM_LOG_LEVEL)
                + '" set to "' + str(self.param_value[Config.PARAM_LOG_LEVEL]) + '".'
            )

            if self.param_value[Config.PARAM_DEBUG] == '1':
                self.param_value[Config.PARAM_DEBUG] = True
                lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
            else:
                self.param_value[Config.PARAM_DEBUG] = False
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': CONFIG "' + str(Config.PARAM_DEBUG)
                + '" set to "' + str(self.param_value[Config.PARAM_DEBUG]) + '".'
            )

            if self.param_value[Config.PARAM_DO_PROFILING] == '1':
                self.param_value[Config.PARAM_DO_PROFILING] = True
            else:
                self.param_value[Config.PARAM_DO_PROFILING] = False
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': CONFIG "' + str(Config.PARAM_DO_PROFILING)
                + '" set to "' + str(self.param_value[Config.PARAM_DO_PROFILING]) + '".'
            )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error initializing config file "' + str(config_file)\
                     + '". Exception message ' + str(ex)
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

    def reset_default_config(
            self
    ):
        # This is the only variable that we should change, the top directory
        topdir = self.get_config(param=Config.PARAM_TOPDIR)
        if not os.path.isdir(topdir):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Fatal error initializing config, "' + str(topdir) + '" is not a directory!'
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

        self.set_default_value_if_not_exist(
            param         = Config.PARAM_LOG_LEVEL,
            default_value = Config.DEFAULT_VALUE_LOGLEVEL
        )

        self.set_default_value_if_not_exist(
            param         = Config.PARAM_DO_PROFILING,
            default_value = Config.DEFAULT_VALUE_DO_PROFILING
        )

        #######################################################################
        # Models Stuff
        #######################################################################

        # Where to store model files
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_MODEL_DIR,
            default_value = topdir + '/app.data/models'
        )

        #######################################################################
        # NLP Stuff
        #######################################################################

        # Built-in Word/Stop lists
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_DIR_WORDLIST,
            default_value = topdir + '/nlp.data/wordlist'
        )
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_POSTFIX_WORDLIST,
            default_value = Config.DEFAULT_VALUE_POSTFIX_WORDLIST
        )
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_POSTFIX_STOPWORDS,
            default_value = Config.DEFAULT_VALUE_POSTFIX_STOPWORDS
        )

        # Application Word/Stop lists
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_DIR_APP_WORDLIST,
            default_value = topdir + '/nlp.data/app/chats'
        )
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_POSTFIX_APP_WORDLIST,
            default_value = Config.DEFAULT_VALUE_POSTFIX_APP_WORDLIST
        )
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_POSTFIX_APP_STOPWORDS,
            default_value = Config.DEFAULT_VALUE_POSTFIX_APP_STOPWORDS
        )

        # Synonym lists
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_DIR_SYNONYMLIST,
            default_value = topdir + '/nlp.data/app/chats'
        )
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_NLP_POSTFIX_SYNONYMLIST,
            default_value = Config.DEFAULT_VALUE_POSTFIX_SYNONYMLIST
        )


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config
    )
    print(config.param_value)
    Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config
    )
    config = Config(
        config_file = '/usr/local/git/nwae/nwae/app.data/config/nwae.cf.local'
    )
    import time
    while True:
        time.sleep(3)
        print(config.get_config(param='topdir'))
        print(config.param_value)