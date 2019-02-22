# -*- coding: utf-8 -*-

#
# We put all our common file paths here
#
class ConfigFile:

    #######################################################################
    # NLP Stuff
    #######################################################################

    # This is the only variable that we should change, the top directory
    TOP_DIR = '/Users/mark.tan/svn/cai.nlp'

    # Word lists
    DIR_WORDLIST     = TOP_DIR + '/nlp.data/wordlist'
    DIR_APP_WORDLIST = TOP_DIR + '/nlp.data/app/chats'
    POSTFIX_WORDLIST = '-wordlist.txt'
    POSTFIX_APP_WORDLIST = '.wordlist.app.txt'

    # Synonym lists
    DIR_SYNONYMLIST = TOP_DIR + '/nlp.data/app/chats'
    POSTFIX_SYNONYMLIST = '.synonymlist.txt'

    # Stopwords lists (to be outdated)
    DIR_APP_STOPWORDS = DIR_APP_WORDLIST
    POSTFIX_APP_STOPWORDS = '.stopwords.app.txt'

    # Various general text for General NLP Training
    DIR_NLP_LANGUAGE_TRAINDATA  = TOP_DIR + '/nlp.data/traindata'

    # Language Stats - Collocation
    DIR_NLP_LANGUAGE_STATS_COLLOCATION = TOP_DIR + '/nlp.output/collocation.stats'

    #######################################################################
    # Chat Analysis Stuff
    #######################################################################

    # Chat Data
    DIR_CHATDATA = TOP_DIR + '/app.data/chatdata'

    # Chat Clustering
    DIR_CHATCLUSTERING_OUTPUT = TOP_DIR + '/app.data/clustering'

    #######################################################################
    # Intent Server Config Stuff
    #######################################################################

    # Intent Training (Contains also, all intents, answers, training data)
    DIR_INTENT_TRAINDATA          = TOP_DIR + '/app.data/intent/traindata'
    DIR_INTENT_TRAIN_LOGS         = DIR_INTENT_TRAINDATA + '/logs'
    POSTFIX_INTENT_ANSWER_TRDATA_FILE = 'chatbot.intent-answer-trdata'
    POSTFIX_INTENT_TRAINING_FILES = 'chatbot.trainingdata'
    # Intent Testing
    DIR_INTENTTEST_TESTDATA        = TOP_DIR + '/app.data/intent/test/'

    # Intent RFV
    DIR_RFV_INTENTS = TOP_DIR + '/app.data/intent/rfv'

    #######################################################################
    # Intent Server
    #######################################################################
    DIR_INTENTSERVER = TOP_DIR + '/app.data/server'
    FILEPATH_INTENTSERVER_LOG = DIR_INTENTSERVER + '/intentserver.log.csv'

    def __init__(self):
        return

    @staticmethod
    def top_dir_changed():
        # Word lists
        ConfigFile.DIR_WORDLIST = ConfigFile.TOP_DIR + '/nlp.data/wordlist'
        ConfigFile.DIR_APP_WORDLIST = ConfigFile.TOP_DIR + '/nlp.data/app/chats'

        # Synonym lists
        ConfigFile.DIR_SYNONYMLIST = ConfigFile.TOP_DIR + '/nlp.data/app/chats'

        # Stopwords lists (to be outdated)
        ConfigFile.DIR_APP_STOPWORDS = ConfigFile.DIR_APP_WORDLIST

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
        # ChatBot Stuff
        #######################################################################

        # Intent Training (Contains also, all intents, answers, training data)
        ConfigFile.DIR_INTENT_TRAINDATA = ConfigFile.TOP_DIR + '/app.data/intent/traindata'
        ConfigFile.DIR_INTENT_TRAIN_LOGS = ConfigFile.DIR_INTENT_TRAINDATA + '/logs'

        # Intent Testing
        ConfigFile.DIR_INTENTTEST_TESTDATA = ConfigFile.TOP_DIR + '/app.data/intent/test/'

        # Intent RFV
        ConfigFile.DIR_RFV_INTENTS = ConfigFile.TOP_DIR + '/app.data/intent/rfv'

        #######################################################################
        # Chat & Bot API Server Stuff
        #######################################################################
        ConfigFile.DIR_INTENTSERVER = ConfigFile.TOP_DIR + '/app.data/server'
        ConfigFile.FILEPATH_INTENTSERVER_LOG = ConfigFile.DIR_INTENTSERVER + '/intentserver.chatlog.csv'
