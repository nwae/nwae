# -*- coding: utf-8 -*-

import os
import nwae.utils.BaseConfig as baseconfig
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# We put all our common file paths here
#
class Config(baseconfig.BaseConfig):

    CONFIG_FILE_PATH_DEFAULT = '/usr/local/git/nwae/nwae.lang/app.data/config/default.cf'

    PARAM_TOPDIR = 'topdir'
    DEFVAL_TOPDIR = '~/nwae'

    PARAM_LOG_LEVEL = 'loglevel'
    DEFVAL_LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    #
    # NLP Settings
    #
    PARAM_NLP_DIR_WORDLIST = 'dir_wordlist'
    DEFVAL_NLP_DIR_WORDLIST = None

    PARAM_NLP_POSTFIX_WORDLIST = 'postfix_wordlist'
    DEFVAL_NLP_POSTFIX_WORDLIST = '-wordlist.txt'

    PARAM_NLP_POSTFIX_STOPWORDS = 'postfix_stopwords'
    DEFVAL_NLP_POSTFIX_STOPWORDS = '-stopwords.txt'

    PARAM_NLP_DIR_APP_WORDLIST = 'dir_app_wordlist'
    DEFVAL_NLP_DIR_APP_WORDLIST = None

    PARAM_NLP_POSTFIX_APP_WORDLIST = 'postfix_app_wordlist'
    DEFVAL_NLP_POSTFIX_APP_WORDLIST = '.wordlist.app.txt'

    PARAM_NLP_DIR_APP_STOPWORDS = 'dir_app_stopwords'
    DEFVAL_NLP_DIR_APP_STOPWORDS = None

    PARAM_NLP_POSTFIX_APP_STOPWORDS = 'postfix_app_stopwords'
    DEFVAL_NLP_POSTFIX_APP_STOPWORDS = '.stopwords.app.txt'

    PARAM_NLP_DIR_SYNONYMLIST = 'dir_synonymlist'
    DEFVAL_NLP_DIR_SYNONYMLIST = None

    PARAM_NLP_POSTFIX_SYNONYMLIST = 'postfix_synonymlist'
    DEFVAL_NLP_POSTFIX_SYNONYMLIST = '.synonymlist.txt'

    # NLTK or whatever download dir
    PARAM_NLP_DIR_NLP_DOWNLOAD = 'nlp_dir_download'
    DEFVAL_NLP_DIR_NLP_DOWNLOAD = None

    #
    # General debug, profiling settings
    #
    PARAM_DEBUG = 'debug'
    DEFVAL_DEBUG = False

    PARAM_DO_PROFILING = 'do_profiling'
    DEFVAL_DO_PROFILING = False

    def __init__(
            self,
            config_file,
            obfuscate_passwords = False
    ):
        super(Config, self).__init__(
            config_file = config_file,
            obfuscate_passwords = obfuscate_passwords
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

            self.__reconfigure_paths_requiring_topdir()

            #
            # This is the part we convert our values to desired types
            #
            self.convert_value_to_float_type(
                param = Config.PARAM_LOG_LEVEL,
                default_val = Config.DEFVAL_LOGLEVEL
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
            lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = self.param_value[Config.PARAM_DEBUG]
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Set debug print to screen to "' + str(lg.Log.DEBUG_PRINT_ALL_TO_SCREEN) + '".'
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

        param_values_to_set = {
            # Top directory
            Config.PARAM_TOPDIR: Config.DEFVAL_TOPDIR,
            # Logging
            Config.PARAM_LOG_LEVEL: Config.DEFVAL_LOGLEVEL,
            #######################################################################
            # Intent Server Stuff
            #######################################################################
            Config.PARAM_DO_PROFILING: Config.DEFVAL_DO_PROFILING,
            #######################################################################
            # NLP Stuff
            #######################################################################
            # Word lists
            Config.PARAM_NLP_DIR_WORDLIST: Config.DEFVAL_NLP_DIR_WORDLIST,
            Config.PARAM_NLP_DIR_APP_WORDLIST: Config.DEFVAL_NLP_DIR_APP_WORDLIST,
            Config.PARAM_NLP_POSTFIX_WORDLIST: Config.DEFVAL_NLP_POSTFIX_WORDLIST,
            Config.PARAM_NLP_POSTFIX_APP_WORDLIST: Config.DEFVAL_NLP_POSTFIX_APP_WORDLIST,
            # Synonym lists
            Config.PARAM_NLP_DIR_SYNONYMLIST: Config.DEFVAL_NLP_DIR_SYNONYMLIST,
            Config.PARAM_NLP_POSTFIX_SYNONYMLIST: Config.DEFVAL_NLP_POSTFIX_SYNONYMLIST,
            # Stopwords lists (to be outdated)
            Config.PARAM_NLP_POSTFIX_STOPWORDS: Config.DEFVAL_NLP_POSTFIX_STOPWORDS,
            Config.PARAM_NLP_DIR_APP_STOPWORDS: Config.DEFVAL_NLP_DIR_APP_STOPWORDS,
            Config.PARAM_NLP_POSTFIX_APP_STOPWORDS: Config.DEFVAL_NLP_POSTFIX_APP_STOPWORDS,
            # NLTK or whatever download dir
            Config.PARAM_NLP_DIR_NLP_DOWNLOAD: Config.DEFVAL_NLP_DIR_NLP_DOWNLOAD,
        }

        for param in param_values_to_set.keys():
            default_value = param_values_to_set[param]
            self.set_default_value_if_not_exist(
                param = param,
                default_value = default_value
            )
        return

    def __reconfigure_paths_requiring_topdir(self):
        topdir = self.get_config(param=Config.PARAM_TOPDIR)

        # Word lists
        if self.param_value[Config.PARAM_NLP_DIR_WORDLIST] is None:
            self.param_value[Config.PARAM_NLP_DIR_WORDLIST] = topdir + '/nlp.data/wordlist'
        if self.param_value[Config.PARAM_NLP_DIR_APP_WORDLIST] is None:
            self.param_value[Config.PARAM_NLP_DIR_APP_WORDLIST] = topdir + '/nlp.data/app/chats'

        # Synonym lists
        if self.param_value[Config.PARAM_NLP_DIR_SYNONYMLIST] is None:
            self.param_value[Config.PARAM_NLP_DIR_SYNONYMLIST] = topdir + '/nlp.data/app/chats'

        # Stopwords lists (to be outdated)
        if self.param_value[Config.PARAM_NLP_DIR_APP_STOPWORDS] is None:
            self.param_value[Config.PARAM_NLP_DIR_APP_STOPWORDS] =\
                self.param_value[Config.PARAM_NLP_DIR_APP_WORDLIST]

        # NLTK or whatever download dir
        if self.param_value[Config.PARAM_NLP_DIR_NLP_DOWNLOAD] is None:
            self.param_value[Config.PARAM_NLP_DIR_NLP_DOWNLOAD] = topdir + '/nlp.data/download'

        return


if __name__ == '__main__':
    import time

    cffile = Config.CONFIG_FILE_PATH_DEFAULT

    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = cffile
    )
    print(config.param_value)
    #
    # Singleton should already exist
    #
    print('*************** Test Singleton exists')
    Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = cffile
    )
    time.sleep(3)

    print('*************** Test config file reload..')
    config = Config(
        config_file = cffile
    )
    while True:
        time.sleep(3)
        if config.is_file_last_updated_time_is_newer():
            print('********************* FILE TIME UPDATED...')
            config.reload_config()
            print(config.get_config(param='topdir'))
            print(config.param_value)