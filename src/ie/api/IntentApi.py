# -*- coding: utf-8 -*-

import flask
from flask import abort
import sys
import ie.app.ConfigFile as cf
import mozg.common.util.Log as lg
import mozg.common.data.security.Auth as au
import datetime as dt
import threading
import numpy as np
import os
from inspect import currentframe, getframeinfo
import ie.lib.server.thread.BotStartupThread as botStartupThread


#
# Flask is not multithreaded, all requests are lined up. This explains why request
# variables are global.
# To make it multithreaded, we declare this app application that already implements
# the method required by the WSGI (gunicorn)
#
app = flask.Flask(__name__)

def Start_Intent_Engine():
    obj = IntentApi()
    return obj

#
# TODO Flask current DOES NOT run in multithreaded mode and handle 1 request at
# TODO one time. Need to use gunicorn or WSGI.
#
class IntentApi:

    # When requests per sec hits this value, we don't lookup DB, but only use cache
    # RPS is measured in the last 5 seconds
    MAX_RPS_BEFORE_USE_ONLY_CACHE = 3
    RPS_MEASURE_LAST_X_SECONDS = 5

    # A reference date for multipurpuse use
    HARDDATE = dt.datetime(2019, 1, 1)

    def __startup(self):
        #
        # Run like '/usr/local/bin/python3.6 -m ie.app.chatbot.server.BotServer brand=fun88 lang=cn'
        #
        # Default values
        pv = {
            'topdir': None,
            'account': '',
            'port': 5000,
            'protocol': 'http',
            'training': '0',
            'minimal': '0',
            'debug': '0',
            'loglevel': lg.Log.LOG_LEVEL_INFO
        }
        args = sys.argv
        usage_msg = 'Usage: ./run.intentapi.sh topdir=/Users/mark.tan/git/mozg.nlp account="Welton" port=5000'

        for arg in args:
            arg_split = arg.split('=')
            if len(arg_split) == 2:
                param = arg_split[0].lower()
                value = arg_split[1]
                if param in list(pv.keys()):
                    pv[param] = value

        if (pv['topdir'] is None):
            errmsg = usage_msg
            raise (Exception(errmsg))

        #
        # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
        #
        cf.ConfigFile.TOP_DIR = pv['topdir']
        cf.ConfigFile.top_dir_changed()

        # If using DB, need to know which Account
        self.db_account_name_to_startup = pv['account']

        # Minimal RAM
        self.minimal = False
        if pv['minimal'] == '1':
            self.minimal = True

        # Can accept training?
        self.accept_training_requests = False
        if pv['training'] == '1':
            self.accept_training_requests = True

        # Logs
        self.loglevel = float(pv['loglevel'])
        lg.Log.set_path(cf.ConfigFile.FILEPATH_INTENTSERVER_LOG)
        lg.Log.LOGLEVEL = self.loglevel
        if pv['debug'] == '1':
            lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                   + ': DB PROFILE ' + cf.ConfigFile.DB_PROFILE
                   + '** INTENT API SERVER STARTUP. Using the following parameters..'
                   + str(pv) + '.')

        self.port = int(pv['port'])
        self.protocol = pv['protocol']

        # DB Stuff
        au.Auth.init_instances()
        return

    def __init_rest_urls(self):
        @self.app.route('/intent', methods=['POST'])
        def intent_api():
            account_id = self.get_param(param_name='accid', method='POST')
            bot_id = self.get_param(param_name='botid', method='POST')
            return self.intent_api_get(account_id=account_id, bot_id=bot_id, method='POST')
        #
        # Intent API: cn
        #
        @self.app.route('/', methods=['GET'])
        def gbot_intent_api_cn():
            account_id = self.get_param(param_name='accid', method='GET')
            bot_id = self.get_param(param_name='botid', method='GET')
            return self.intent_api_get(account_id=account_id, bot_id=bot_id)

        #
        # Segment Words API: cn
        # TODO Remove brand dependence
        #
        @self.app.route('/segmentwords', methods=['GET'])
        def gbot_api_segment_words_cn_fun88():
            account_id = self.get_param(param_name='accid', method='GET')
            bot_id = self.get_param(param_name='botid', method='GET')
            return self.segment_words_api_get(account_id=account_id, bot_id=bot_id)

        #
        # Train Bot API: cn, fun88
        #
        @self.app.route('/trainbot', methods=['GET'])
        def api_train_bot():
            account_id = self.get_param(param_name='accid', method='GET')
            bot_id = self.get_param(param_name='botid', method='GET')
            return self.train_bot(lang='cn', account_id=account_id, bot_id=bot_id)

        @self.app.route('/restartbots', methods=['GET'])
        def api_restart_bots():
            return self.restart_bots()

        @self.app.errorhandler(404)
        def page_not_found(e):
            lg.Log.error(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                       + ': Resource [' + str(flask.request.url) + '] is not valid!')
            return "<h1>404</h1><p>The resource could not be found.</p>", 404

    def __init__(
            self
    ):
        self.__startup()

        self.app = app
        self.app.config['DEBUG'] = False

        try:
            import psutil
            self.worker_name = str(psutil.Process())
        except Exception as ex:
            self.worker_name = str(dt.datetime.now())
            lg.Log.error(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Worker name error: ' + str(ex) + '.')
        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Worker "' + self.worker_name + '" starting...')

        # To keep track of RPS
        self.request_per_second = 0
        self.request_times = np.array([])
        self.__request_mutex = threading.Lock()
        self.request_total = 0

        self.bot_startup_thread = None
        # This moves into background thread without slowing down the initialization of Rest URLs
        # This is critical so that gunicorn do not keep restarting the app that takes too long
        self.restart_bots()

        self.__init_rest_urls()
        return

    def record_request(self):
        td = (dt.datetime.now() - IntentApi.HARDDATE)
        td_secs = float( td.days*24*60*60 + td.seconds + td.microseconds/1000000 )

        self.__request_mutex.acquire()
        self.request_times = np.append(arr=self.request_times, values=td_secs)
        self.__request_mutex.release()

    def is_rps_exceed_limit(
            self
    ):
        self.__request_mutex.acquire()

        td = (dt.datetime.now() - IntentApi.HARDDATE)
        now_td_secs = float( td.days*24*60*60 + td.seconds + td.microseconds/1000000 )

        self.request_times = self.request_times[
            (now_td_secs - self.request_times <= IntentApi.RPS_MEASURE_LAST_X_SECONDS)]

        self.request_per_second = len(self.request_times) / IntentApi.RPS_MEASURE_LAST_X_SECONDS
        self.__request_mutex.release()

        return (self.request_per_second > IntentApi.MAX_RPS_BEFORE_USE_ONLY_CACHE)

    def intent_api_get(
            self,
            account_id,
            bot_id,
            method = 'GET'
    ):
        self.record_request()
        use_only_cache_data = self.is_rps_exceed_limit()

        if use_only_cache_data:
            lg.Log.warning(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                       + ': WORKER [' + str(self.worker_name)
                       + '] RPS WARNING, RPS ' + str(self.request_per_second)
                       + ' >= ' + str(IntentApi.MAX_RPS_BEFORE_USE_ONLY_CACHE)
                       + '. Using only cache data for now until RPS reduces..')

        self.request_total = self.request_total + 1
        lg.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                   + ': WORKER [' + str(self.worker_name) + '] TOTAL REQUESTS = ' + str(self.request_total)
                   + ', RPS ' + str(self.request_per_second) + ' in the last '
                   + str(IntentApi.RPS_MEASURE_LAST_X_SECONDS) + ' seconds.')

        s = self.get_param(param_name='txt', method=method)
        chatid = self.get_param(param_name='chatid', method=method)

        info_msg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                   + ': Query=Intent, Account ID=' + str(account_id) + ', Bot ID=' + str(bot_id) \
                   + ', ChatId=' + str(chatid) + ', Intent query txt=' + str(s) + '.'
        lg.Log.important(info_msg)

        if s is None:
            return "Error: No txt field provided. Please specify a txt."
        else:
            try:
                answer_json = self.bot_startup_thread.handle_intent_request(
                    account_id = account_id,
                    bot_id     = bot_id,
                    question   = s,
                    chatid     = chatid,
                    use_only_cache_data = use_only_cache_data
                )
                return answer_json
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                           + ': Exception occurred for [' + info_msg + ']: ' + str(ex)
                lg.Log.critical(errmsg)
                if lg.Log.DEBUG_PRINT_ALL_TO_SCREEN:
                    raise Exception(errmsg)
                abort(500)

    def segment_words_api_get(
            self,
            account_id,
            bot_id
    ):
        text = self.get_param(param_name='txt')
        lg.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': IP ' + str(flask.request.remote_addr)
                         +', Query=Segment Words, Account ID =' + account_id
                         + ', Bot ID=' + bot_id + ', Txt=' + str(text) + '.')

        if text is None:
            return "Error: No txt field provided. Please specify a txt."
        else:
            try:
                return self.bot_startup_thread.handle_segment_words_request(
                    account_id = account_id,
                    bot_id     = bot_id,
                    text       = text
                )
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                         + ' Exception occurred segment words for IP ' + str(flask.request.remote_addr)\
                         + ', text "' + str(text) + '", exception ' + str(ex) + '.'
                lg.Log.critical(errmsg)
                if lg.Log.DEBUG_PRINT_ALL_TO_SCREEN:
                    raise Exception(errmsg)
                return 'Sorry, could not get answer'

    def train_bot(
            self,
            account_id,
            bot_id,
            lang
    ):
        try:
            msg = self.bot_startup_thread.handle_train_bot_request(
                account_id = account_id,
                bot_id     = bot_id
            )
            return msg
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception [' + str(ex) + '] training bot for account id '\
                     + str(account_id) + ', bot ID ' + str(bot_id) + ', lang ' + str(lang) + '.'
            lg.Log.critical(errmsg)
            raise errmsg

    def restart_bots(
            self
    ):
        if self.bot_startup_thread is not None:
            self.bot_startup_thread.join(timeout=2)

        lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Starting new bot thread')

        self.bot_startup_thread = botStartupThread.BotStartupThread(
            account_name_to_startup   = self.db_account_name_to_startup,
            minimal                   = self.minimal,
            use_db                    = cf.ConfigFile.USE_DB,
            db_profile                = cf.ConfigFile.DB_PROFILE,
            cf_dir_rfv_commands       = cf.ConfigFile.DIR_RFV_INTENTS,
            cf_dir_synonymlist        = cf.ConfigFile.DIR_SYNONYMLIST,
            cf_dir_wordlist           = cf.ConfigFile.DIR_WORDLIST,
            cf_postfix_wordlist       = cf.ConfigFile.POSTFIX_WORDLIST,
            cf_dir_wordlist_app       = cf.ConfigFile.DIR_APP_WORDLIST,
            cf_postfix_wordlist_app   = cf.ConfigFile.POSTFIX_APP_WORDLIST,
            cf_dirpath_traindata      = cf.ConfigFile.DIR_INTENT_TRAINDATA,
            cf_postfix_training_files = cf.ConfigFile.POSTFIX_INTENT_TRAINING_FILES,
            do_profiling              = True,
            accept_training_requests  = self.accept_training_requests
        )
        self.bot_startup_thread.start()
        return 'Bots restarting'

    def get_param(self, param_name, method='GET'):
        if method == 'GET':
            if param_name in flask.request.args:
                return str(flask.request.args[param_name])
            else:
                return ''
        else:
            try:
                val = flask.request.json[param_name]
                return val
            except Exception as ex:
                lg.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                           + ': No param name [' + param_name + '] in request.')
                return None

    def run(self, host='0.0.0.0'):
        if self.protocol == 'https':
            self.app.run(host=host, port=self.port, ssl_context='adhoc')
        else:
            self.app.run(
                host      = host,
                port      = self.port,
                # threaded = True
            )


if __name__ == '__main__':
    intent_api_instance = Start_Intent_Engine()
    intent_api_instance.run()
