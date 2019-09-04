# -*- coding: utf-8 -*-

import os
import sys
import nwae.utils.StringUtils as su
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# We put all our common file paths here
#
class ConfigFile:

    DEFAULT_LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    SINGLETON = None

    #
    # Always call this method only to make sure we get singleton
    #
    @staticmethod
    def get_cmdline_params_and_init_config_singleton():
        if type(ConfigFile.SINGLETON) is ConfigFile:
            lg.Log.info(
                str(ConfigFile.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Config Singleton from file "' + str(ConfigFile.SINGLETON.CONFIGFILE)
                + '" exists. Returning Singleton..'
            )
            return ConfigFile.SINGLETON

        # Default values
        pv = {
            'configfile': None
        }
        args = sys.argv

        for arg in args:
            arg_split = arg.split('=')
            if len(arg_split) == 2:
                param = arg_split[0].lower()
                value = arg_split[1]
                if param in list(pv.keys()):
                    pv[param] = value

        if pv['configfile'] is None:
            raise Exception('"configfile" param not found on command line!')

        #
        # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
        #
        ConfigFile.SINGLETON = ConfigFile(
            config_file = pv['configfile']
        )
        return ConfigFile.SINGLETON

    def __init__(
            self,
            config_file
    ):
        self.CONFIGFILE = config_file
        if not os.path.isfile(self.CONFIGFILE):
            raise Exception('Configfile "' + str(self.CONFIGFILE) + '" is not a valid file path!')

        # Default values
        pv = {
            'topdir': None,
            'debug': '0',
            'do_profiling': '0',
            'loglevel': ConfigFile.DEFAULT_LOGLEVEL
        }
        try:
            f = open(config_file, 'r')
            linelist_file = f.readlines()
            f.close()

            linelist = []
            for line in linelist_file:
                line = su.StringUtils.trim(su.StringUtils.remove_newline(line))
                # Ignore comment lines
                if line[0] == '#':
                    continue
                linelist.append(line)

            for line in linelist:
                arg_split = line.split('=')
                if len(arg_split) == 2:
                    param = arg_split[0].lower()
                    value = arg_split[1]
                    if param in list(pv.keys()):
                        pv[param] = value

            lg.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Read from app config file "' + str(config_file)
                + ', file lines:\n\r' + str(linelist) + ', properties\n\r' + str(pv)
            )

            self.TOP_DIR = pv['topdir']
            self.reset_default_config()

            self.LOGLEVEL = float(pv['loglevel'])
            lg.Log.critical(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': CONFIG LOGLEVEL set to "' + str(self.LOGLEVEL) + '".'
            )
            lg.Log.LOGLEVEL = float(pv['loglevel'])
            if pv['debug'] == '1':
                lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
                lg.Log.critical(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': CONFIG DEBUG_PRINT_ALL_TO_SCREEN set to "' + str(lg.Log.DEBUG_PRINT_ALL_TO_SCREEN) + '".'
                )

            if pv['do_profiling'] == '1':
                self.DO_PROFILING = True
                lg.Log.critical(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': CONFIG DO_PROFILING set to "' + str(self.DO_PROFILING) + '".'
                )
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error reading app config file "' + str(config_file)\
                     + '". Exception message ' + str(ex)
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

    def reset_default_config(
            self
    ):
        # This is the only variable that we should change, the top directory
        if not os.path.isdir(self.TOP_DIR):
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Fatal error initializing config, "' + str(self.TOP_DIR) + '" is not a directory!'
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

        self.LOGLEVEL = ConfigFile.DEFAULT_LOGLEVEL
        self.DO_PROFILING = False

        #######################################################################
        # Models Stuff
        #######################################################################

        # Where to store model files
        self.DIR_MODELS = self.TOP_DIR + '/app.data/models'

        #######################################################################
        # NLP Stuff
        #######################################################################

        # Word lists
        self.DIR_WORDLIST = self.TOP_DIR + '/nlp.data/wordlist'
        self.DIR_APP_WORDLIST = self.TOP_DIR + '/nlp.data/app/chats'
        self.POSTFIX_WORDLIST = '-wordlist.txt'
        self.POSTFIX_APP_WORDLIST = '.wordlist.app.txt'
        self.POSTFIX_STOPWORDS = '-stopwords.txt'

        # Synonym lists
        self.DIR_SYNONYMLIST = self.TOP_DIR + '/nlp.data/app/chats'
        self.POSTFIX_SYNONYMLIST = '.synonymlist.txt'

        # Stopwords lists (to be outdated)
        self.DIR_APP_STOPWORDS = self.DIR_APP_WORDLIST
        self.POSTFIX_APP_STOPWORDS = '.stopwords.app.txt'

        # Various general text for General NLP Training
        self.DIR_NLP_LANGUAGE_TRAINDATA = self.TOP_DIR + '/nlp.data/traindata'

        # Language Stats - Collocation
        self.DIR_NLP_LANGUAGE_STATS_COLLOCATION = self.TOP_DIR + '/nlp.output/collocation.stats'
