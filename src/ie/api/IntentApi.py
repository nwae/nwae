# -*- coding: utf-8 -*-

import flask
import sys
import ie.app.ConfigFile as cf
import ie.lib.chat.bot.LeBotWrapper as gb
import ie.lib.util.Log as lg
import datetime as dt
import time as t
import json


class GBotApi:

    BOTS_TO_INITIALIZE = [['cn','fun88'], ['th','fun88']]

    def __init__(
            self,
            port,
            verbose=0
    ):
        self.port = port

        self.app = flask.Flask(__name__)
        self.app.config['DEBUG'] = False

        self.bots = {}

        for lb in GBotApi.BOTS_TO_INITIALIZE:
            lang = lb[0]
            brand = lb[1]
            botkey = lang + '.' + brand
            self.bots[botkey] = self.get_bot(lang=lang, brand=brand)

        @self.app.route('/cn/fun88', methods=['GET'])
        def gbot_api_cn_fun88():
            return self.api_get(lang='cn', brand='fun88')

        @self.app.route('/th/fun88', methods=['GET'])
        def gbot_api_th_fun88():
            return self.api_get(lang='th', brand='fun88')

        @self.app.errorhandler(404)
        def page_not_found(e):
            lg.Log.log('Resource [' + str(flask.request.url) + '] is not valid!')
            return "<h1>404</h1><p>The resource could not be found.</p>", 404

    def api_get(self, lang, brand):
        ts = dt.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d %H:%M:%S')
        addr = str(flask.request.remote_addr)
        s = self.get_txt()

        info_msg = 'Lang=' + lang + ', Brand=' + brand + ', Query txt=' + s + '.'
        lg.Log.log(str(ts) + ': IP=' + addr + ', ' + info_msg)
        if s is None:
            return "Error: No txt field provided. Please specify a txt."
        else:
            botkey = lang + '.' + brand
            try:
                df_com_class = self.bots[botkey].get_text_class(inputtext=s, top=5)
                answer_json = self.bots[botkey].get_text_answer(df_intent=df_com_class, reply_format='json')

                #load_back = json.loads(answer_json)
                #lg.Log.log('Loaded back as ' + str(load_back))

                if answer_json is None:
                    return json.dumps({})
                else:
                    return answer_json
            except Exception as ex:
                lg.Log.log(str(self.__class__) + ' Exception occurred for [' + info_msg + ']')
                lg.Log.log(ex)
                return 'Sorry, could not get answer'

    def get_bot(self, lang, brand):
        lg.Log.log('Initializing Bot for ' + lang + '.' + brand + '.')
        bot = gb.LeBotWrapper(
            lang             = lang,
            brand            = brand,
            dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
            dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
            dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix_wordlist_app = '.' + brand + cf.ConfigFile.POSTFIX_APP_WORDLIST
        )
        bot.init()
        return bot

    def get_txt(self):
        if 'txt' in flask.request.args:
            return str(flask.request.args['txt'])
        else:
            return ''

    def run(self, host='0.0.0.0'):
        self.app.run(host=host, port=self.port)


if __name__ == '__main__':
    #
    # Run like '/usr/local/bin/python3.6 -m ie.app.chatbot.server.BotServer brand=fun88 lang=cn'
    #
    # Default values
    pv = {
        'topdir': None,
        'port': 5000,
        'debug': '0',
        'verbose': '1'
    }
    args = sys.argv
    usage_msg = 'Usage: ./run.intentapi.sh topdir=/home/mark/svn/cai.nlp'

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

    # Logs
    lg.Log.set_path(cf.ConfigFile.FILEPATH_INTENTSERVER_LOG)
    if pv['debug'] == '1':
        lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    lg.Log.log('BOT API SERVER STARTUP. Using the following parameters..')
    lg.Log.log(str(pv))

    gbotapi = GBotApi(port=int(pv['port']))
    gbotapi.run()