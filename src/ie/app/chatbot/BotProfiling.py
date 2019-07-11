# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import ie.lib.chat.bot.IntentWrapper as inttwrp
import mozg.common.util.Log as lg
import mozg.common.util.Db as dbutil
import mozg.common.util.Profiling as prf
import mozg.common.data.Campaign as dbcp
import mozg.common.data.security.Auth as au
import sys
import json
import mozg.common.util.CommandLine as cmdline


class BotProfiling:

    PROFILING_TEXTS = [
        '提款',
        '提款存款密码游戏',
        '吃饭了没有',
        # Check if cache intents is working or not
        '提款存款密码游戏',
    ]

    def __init__(
            self,
            account_id,
            bot_id,
            bot_lang,
            botkey,
            reduce_features
    ):
        self.db_account_id = account_id
        self.db_bot_id = bot_id
        self.bot_lang = bot_lang
        self.botkey = botkey
        self.reduce_features = reduce_features

    def run(
            self,
            verbose = 1
    ):

        prfbot = inttwrp.IntentWrapper(
            use_db      = cf.ConfigFile.USE_DB,
            db_profile  = cf.ConfigFile.DB_PROFILE,
            account_id  = self.db_account_id,
            bot_id      = self.db_bot_id,
            lang        = self.bot_lang,
            bot_key     = self.botkey,
            dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
            dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
            dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST,
            do_profiling         = True,
            min_score_threshold  = 5,
            verbose              = verbose
        )
        prfbot.init()

        for chatstr in BotProfiling.PROFILING_TEXTS:
            lg.Log.log('PROFILING [' + chatstr + ']...')

            start_intent = prf.Profiling.start()
            lg.Log.log('.  PROFILING Intent Start: ' + str(start_intent))
            # Get Intent(s)
            df_com_class = prfbot.get_text_class(
                chatid    = None,
                inputtext = chatstr,
                reduced_features = self.reduce_features
            )
            end_intent = prf.Profiling.stop()
            lg.Log.log('.  PROFILING Intent End: ' + prf.Profiling.get_time_dif_str(start=start_intent, stop=end_intent))

            start_intent_txt = prf.Profiling.start()
            lg.Log.log('.  PROFILING Intent Format Answer Start: ' + str(start_intent_txt))
            # TODO Format answer into text (this is the part that is TOO SLOW)
            json_reply = prfbot.get_json_response(
                chatid    = None,
                df_intent = df_com_class
            )
            end_intent_txt = prf.Profiling.stop()
            lg.Log.log('.  PROFILING Intent Format Answer Time: ' + prf.Profiling.get_time_dif_str(
                start=start_intent_txt, stop=end_intent_txt))

            if verbose >= 1:
                dt = json.loads(json_reply)
                dt_matches = dt['matches']
                for key in dt_matches.keys():
                    s = str(key) + '. ' + str(dt_matches[key])
                    lg.Log.log(s)

        return


if __name__ == '__main__':

    #
    # Run like '/usr/local/bin/python3.6 -m ie.app.chatbot.server.BotServer brand=fun88 lang=cn'
    #
    # Default values
    pv = {
        'topdir': None,
        'account': 'Welton',
        'campaign': 'Fun88 CNY',
        'reduceFeatures': False,
        'verbose': '1'
    }
    args = sys.argv
    usage_msg = 'Usage: ./run.intentapi.sh topdir=/Users/mark.tan/git/mozg.nlp account=Welton campaign=...'

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
    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile=cf.ConfigFile.DB_PROFILE
    )
    bot = BotProfiling(
        account_id = accountId,
        bot_id     = botId,
        bot_lang   = botLang,
        botkey     = botkey,
        reduce_features = pv['reduceFeatures']
    )

    bot.run()
