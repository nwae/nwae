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

    #
    # General debug, profiling settings
    #
    PARAM_DEBUG = 'debug'
    DEFAULT_VALUE_DEBUG = False

    PARAM_DO_PROFILING = 'do_profiling'
    DEFAULT_VALUE_DO_PROFILING = False

    #
    # Model Back Testing
    #
    PARAM_MODEL_BACKTEST_DETAILED_STATS = 'model_backtest_detailed_stats'
    DEFAULT_VALUE_MODEL_BACKTEST_DETAILED_STATS = True

    def __init__(
            self,
            config_file
    ):
        super(Config, self).__init__(
            config_file = config_file
        )
        self.reload_config()
        return

    def reload_config(
            self
    ):
        # Call base class first
        lg.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Calling base class reload config for "' + str(self.config_file) + '"..'
        )
        super(Config,self).reload_config()

        try:
            self.reset_default_config()

            #
            # This is the part we convert our values to desired types
            #
            self.convert_value_to_float_type(
                param = Config.PARAM_LOG_LEVEL,
                default_val = Config.DEFAULT_VALUE_LOGLEVEL
            )
            lg.Log.LOGLEVEL = self.param_value[Config.PARAM_LOG_LEVEL]
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Set log level to "' + str(lg.Log.LOGLEVEL) + '".'
            )

            #
            # Here lies the important question, should we standardize all config
            # to only string type, or convert them?
            #
            self.convert_value_to_boolean_type(
                param = Config.PARAM_DEBUG
            )

            self.convert_value_to_boolean_type(
                param = Config.PARAM_DO_PROFILING
            )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error initializing config file "' + str(self.config_file)\
                     + '". Exception message ' + str(ex)
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

        return

    #
    # For those params not found in config file, we give default values
    #
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

        #
        # Model Backtesting
        #
        self.set_default_value_if_not_exist(
            param         = Config.PARAM_MODEL_BACKTEST_DETAILED_STATS,
            default_value = Config.DEFAULT_VALUE_MODEL_BACKTEST_DETAILED_STATS
        )
        return


if __name__ == '__main__':
    import time

    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config
    )
    print(config.param_value)
    #
    # Singleton should already exist
    #
    print('*************** Test Singleton exists')
    Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config
    )
    time.sleep(3)

    print('*************** Test config file reload..')
    config = Config(
        config_file = '/usr/local/git/nwae/nwae/app.data/config/nwae.cf.local'
    )
    while True:
        time.sleep(3)
        if config.is_file_last_updated_time_is_newer():
            print('********************* FILE TIME UPDATED...')
            config.reload_config()
            print(config.get_config(param='topdir'))
            print(config.param_value)