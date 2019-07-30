# -*- coding: utf-8 -*-

import os
import mozg.common.util.StringUtils as su
import mozg.common.util.Log as lg
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

    #######################################################################
    # DB Stuff (not dependent on topdir)
    #######################################################################
    # DB
    USE_DB = None
    # This is the database to connect to, that might contain all account info
    DB_PROFILE = None

    #######################################################################
    # Intent Server Stuff
    #######################################################################
    PORT                     = None
    RUNMODE_STAGING          = None
    BOT_IDS_TO_STARTUP       = None
    ACCEPT_TRAINING_REQUESTS = None
    MINIMAL_SERVER           = None
    DO_PROFILING             = None

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

    #######################################################################
    # Chat Analysis Stuff
    #######################################################################

    # Chat Data
    DIR_CHATDATA = None

    # Chat Clustering
    DIR_CHATCLUSTERING_OUTPUT = None

    #######################################################################
    # Intent Server Config Stuff
    #######################################################################

    # Intent Training (Contains also, all intents, answers, training data)
    DIR_INTENT_TRAINDATA    = None
    DIR_INTENT_TRAIN_LOGS   = None
    POSTFIX_INTENT_ANSWER_TRDATA_FILE = None
    POSTFIX_INTENT_TRAINING_FILES     = None

    # Intent Testing
    DIR_INTENTTEST_TESTDATA = None

    # Intent RFV
    DIR_RFV_INTENTS = None

    #######################################################################
    # Intent Server
    #######################################################################
    DIR_INTENTSERVER = None
    FILEPATH_INTENTSERVER_LOG = None

    #######################################################################
    # General
    #######################################################################
    DIR_GENERAL_APP = None
    FILEPATH_GENERAL_LOG = None

    @staticmethod
    def init_from_app_config_file(
            config_file,
            # Overwrites port in config file
            port = None
    ):
        # Default values
        pv = {
            'topdir': None,
            'runmode_staging': '0',
            'bot_ids': '',
            'port': 5000,
            'training': '0',
            'minimal': '0',
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

            # If using DB, need to know which Account
            ConfigFile.BOT_IDS_TO_STARTUP = pv['bot_ids']

            if int(pv['runmode_staging']) == 0:
                ConfigFile.RUNMODE_STAGING = False
            else:
                ConfigFile.RUNMODE_STAGING = True

            # Minimal RAM
            ConfigFile.MINIMAL_SERVER = False
            if pv['minimal'] == '1':
                ConfigFile.MINIMAL_SERVER = True

            # Can accept training?
            ConfigFile.ACCEPT_TRAINING_REQUESTS = False
            if pv['training'] == '1':
                ConfigFile.ACCEPT_TRAINING_REQUESTS = True

            # Logs
            lg.Log.set_path(ConfigFile.FILEPATH_INTENTSERVER_LOG)
            lg.Log.LOGLEVEL = float(pv['loglevel'])
            if pv['debug'] == '1':
                lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
            lg.Log.critical(
                str(ConfigFile.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': DB PROFILE ' + ConfigFile.DB_PROFILE
                + '** mozg config using the following parameters..'
                + str(pv) + '.'
            )

            if port is not None:
                ConfigFile.PORT = int(port)
            else:
                ConfigFile.PORT = int(pv['port'])

            if int(pv['do_profiling']) == 0:
                ConfigFile.DO_PROFILING = False
            else:
                ConfigFile.DO_PROFILING = True
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
        # DB Stuff (not dependent on topdir)
        #######################################################################
        # DB
        ConfigFile.USE_DB = True
        # This is the database to connect to, that might contain all account info
        ConfigFile.DB_PROFILE = 'mario2'

        #######################################################################
        # Intent Server Stuff
        #######################################################################
        ConfigFile.PORT = 5000
        ConfigFile.RUNMODE_STAGING = False
        ConfigFile.BOT_IDS_TO_STARTUP = ''
        ConfigFile.ACCEPT_TRAINING_REQUESTS = False
        ConfigFile.MINIMAL_SERVER = False
        ConfigFile.DO_PROFILING = True

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

        #######################################################################
        # Chat Analysis Stuff
        #######################################################################

        # Chat Data
        ConfigFile.DIR_CHATDATA = ConfigFile.TOP_DIR + '/app.data/chatdata'

        # Chat Clustering
        ConfigFile.DIR_CHATCLUSTERING_OUTPUT = ConfigFile.TOP_DIR + '/app.data/chat.clustering'

        #######################################################################
        # Intent Server Config Stuff
        #######################################################################

        # Intent Training (Contains also, all intents, answers, training data)
        ConfigFile.DIR_INTENT_TRAINDATA = ConfigFile.TOP_DIR + '/app.data/intent/traindata'
        ConfigFile.DIR_INTENT_TRAIN_LOGS = ConfigFile.DIR_INTENT_TRAINDATA + '/logs'
        ConfigFile.POSTFIX_INTENT_ANSWER_TRDATA_FILE = 'chatbot.intent-answer-trdata'
        ConfigFile.POSTFIX_INTENT_TRAINING_FILES = 'chatbot.trainingdata'

        # Intent Testing
        ConfigFile.DIR_INTENTTEST_TESTDATA = ConfigFile.TOP_DIR + '/app.data/intent/test/'

        # Intent RFV
        ConfigFile.DIR_RFV_INTENTS = ConfigFile.TOP_DIR + '/app.data/intent/rfv'

        #######################################################################
        # Intent Server
        #######################################################################
        ConfigFile.DIR_INTENTSERVER = ConfigFile.TOP_DIR + '/app.data/server'
        ConfigFile.FILEPATH_INTENTSERVER_LOG = ConfigFile.DIR_INTENTSERVER + '/intentserver.log'

        #######################################################################
        # General
        #######################################################################
        ConfigFile.DIR_GENERAL_APP = ConfigFile.TOP_DIR + '/app.data/general'
        ConfigFile.FILEPATH_GENERAL_LOG = ConfigFile.DIR_GENERAL_APP + '/intent.general.log'
