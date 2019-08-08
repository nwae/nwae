# -*- coding: utf-8 -*-

import os
import sys
import mozg.utils.StringUtils as su
import mozg.utils.Log as lg
from inspect import currentframe, getframeinfo


#
# We put all our common file paths here
#
class ConfigFile:
    #
    # There is no need to declare the class variables as None, but if we don't
    # the IDE will show warning/error, which is error prone in programming, since
    # we won't know the correct variable name.
    # Thus declaring as None is to prevent programmer error only, and not necessary
    # because Python in realtime will make the variable available after __init_config().
    #

    TOP_DIR = None

    #######################################################################
    # Models Stuff
    #######################################################################

    # Where to store model files
    DIR_MODELS     = None

    #######################################################################
    # NLP Stuff
    #######################################################################

    # Word lists
    DIR_WORDLIST         = None
    DIR_APP_WORDLIST     = None
    POSTFIX_WORDLIST     = None
    POSTFIX_APP_WORDLIST = None

    # Synonym lists
    DIR_SYNONYMLIST      = None
    POSTFIX_SYNONYMLIST  = None

    # Stopwords lists (to be outdated)
    DIR_APP_STOPWORDS     = None
    POSTFIX_APP_STOPWORDS = None

    # Various general text for General NLP Training
    DIR_NLP_LANGUAGE_TRAINDATA = None

    # Language Stats - Collocation
    DIR_NLP_LANGUAGE_STATS_COLLOCATION = None

    @staticmethod
    def get_cmdline_params_and_init_config():
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

        if (pv['configfile'] is None):
            raise (Exception('"configfile" param not found on command line!'))

        #
        # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
        #
        ConfigFile.init_from_app_config_file(
            config_file=pv['configfile']
        )

    @staticmethod
    def init_from_app_config_file(
            config_file,
            # Overwrites port in config file
            port = None
    ):
        # Default values
        pv = {
            'topdir': None,
            'debug': '0',
            'do_profiling': '1',
            'loglevel': lg.Log.LOG_LEVEL_INFO
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
                str(ConfigFile.__name__) + str(getframeinfo(currentframe()).lineno)
                + ': Read from app config file "' + str(config_file)
                + ', file lines:\n\r' + str(linelist) + ', properties\n\r' + str(pv)
                + '\n\r, port override = ' + str(port) + '.'
            )

            #
            # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
            #
            ConfigFile.init_config(
                topdir = pv['topdir']
            )

            lg.Log.LOGLEVEL = float(pv['loglevel'])
        except Exception as ex:
            errmsg = str(ConfigFile.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Error reading app config file "' + str(config_file)\
                     + '". Exception message ' + str(ex)
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

    @staticmethod
    def init_config(
            topdir
    ):
        # This is the only variable that we should change, the top directory
        ConfigFile.TOP_DIR = topdir
        if not os.path.isdir(topdir):
            errmsg = str(ConfigFile.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Fatal error initializing config, "' + str(topdir) + '" is not a directory!'
            lg.Log.critical(errmsg)
            raise Exception(errmsg)

        #######################################################################
        # Models Stuff
        #######################################################################

        # Where to store model files
        ConfigFile.DIR_MODELS = ConfigFile.TOP_DIR + '/app.data/models'

        #######################################################################
        # NLP Stuff
        #######################################################################

        # Word lists
        ConfigFile.DIR_WORDLIST = ConfigFile.TOP_DIR + '/nlp.data/wordlist'
        ConfigFile.DIR_APP_WORDLIST = ConfigFile.TOP_DIR + '/nlp.data/app/chats'
        ConfigFile.POSTFIX_WORDLIST = '-wordlist.txt'
        ConfigFile.POSTFIX_APP_WORDLIST = '.wordlist.app.txt'

        # Synonym lists
        ConfigFile.DIR_SYNONYMLIST = ConfigFile.TOP_DIR + '/nlp.data/app/chats'
        ConfigFile.POSTFIX_SYNONYMLIST = '.synonymlist.txt'

        # Stopwords lists (to be outdated)
        ConfigFile.DIR_APP_STOPWORDS = ConfigFile.DIR_APP_WORDLIST
        ConfigFile.POSTFIX_APP_STOPWORDS = '.stopwords.app.txt'

        # Various general text for General NLP Training
        ConfigFile.DIR_NLP_LANGUAGE_TRAINDATA = ConfigFile.TOP_DIR + '/nlp.data/traindata'

        # Language Stats - Collocation
        ConfigFile.DIR_NLP_LANGUAGE_STATS_COLLOCATION = ConfigFile.TOP_DIR + '/nlp.output/collocation.stats'
