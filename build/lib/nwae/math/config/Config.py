# -*- coding: utf-8 -*-

import os
import nwae.utils.BaseConfig as baseconfig
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# We put all our common file paths here
#
class Config(baseconfig.BaseConfig):

    CONFIG_FILE_PATH_DEFAULT = '/usr/local/git/nwae/nwae.math/app.data/config/default.cf'

    PARAM_TOPDIR = 'topdir'
    DEFVAL_TOPDIR = '~/nwae'

    PARAM_LOG_LEVEL = 'loglevel'
    DEFVAL_LOGLEVEL = lg.Log.LOG_LEVEL_INFO

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
            obfuscate_passwords = obfuscate_passwords,
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