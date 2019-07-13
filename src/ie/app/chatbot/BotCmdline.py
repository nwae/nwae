# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import mozg.lib.chat.bot.IntentEngineTest as lb
import mozg.lib.chat.bot.IntentWrapper as gb
import mozg.common.util.CommandLine as cmdline
import mozg.common.data.security.Auth as au
import mozg.common.util.Log as lg


class BotCmdline:

    TEXT_PROFILING_1 = '提款存款密码游戏'

    def __init__(self):
        return

    def run(
            self,
            account_id,
            bot_id,
            botkey,
            lang,
            verbose = 1
    ):

        crmbot = gb.IntentWrapper(
            use_db      = cf.ConfigFile.USE_DB,
            db_profile  = cf.ConfigFile.DB_PROFILE,
            account_id  = account_id,
            bot_id      = bot_id,
            lang        = lang,
            bot_key     = botkey,
            dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
            dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
            dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix_wordlist_app = cf.ConfigFile.POSTFIX_APP_WORDLIST,
            do_profiling         = True,
            min_score_threshold  = 5
        )
        crmbot.init()

        crmbot.test_run_on_command_line(
            top                 = lb.IntentEngine.SEARCH_TOPX_RFV,
            verbose             = verbose
        )
        return


if __name__ == '__main__':
    db_profile = cf.ConfigFile.DB_PROFILE

    # DB Stuff initializations
    au.Auth.init_instances()

    [accountId, botId, botLang, botkey] = cmdline.CommandLine.get_parameters_to_run_bot(
        db_profile=cf.ConfigFile.DB_PROFILE
    )

    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    bot = BotCmdline()
    bot.run(
        account_id = accountId,
        bot_id     = botId,
        lang       = botLang,
        botkey     = botkey
    )
