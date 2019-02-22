# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import ie.app.ConfigFile as cf
import ie.lib.chat.bot.Intent as lb
import ie.lib.chat.bot.LeBotWrapper as gb
import ie.app.CommandLine as cmdline


class CRMBot:

    def __init__(self):
        return

    def run(self, lang, brand, verbose=1):

        crmbot = gb.LeBotWrapper(
            lang      = lang,
            brand     = brand,
            dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS,
            dir_synonymlist  = cf.ConfigFile.DIR_SYNONYMLIST,
            dir_wordlist     = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            dir_wordlist_app = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix_wordlist_app = '.' + brand + cf.ConfigFile.POSTFIX_APP_WORDLIST,
            min_score_threshold  = 5
        )
        crmbot.init(verbose=verbose)

        crmbot.run(top=lb.Intent.SEARCH_TOPX_RFV, verbose=verbose)
        return


if __name__ == '__main__':
    ui_lang = None
    ui_brand = None

    while ui_lang is None:
        ui_lang = cmdline.CommandLine.get_user_input_language()

    while ui_brand is None:
        ui_brand = cmdline.CommandLine.get_user_input_brand()

    bot = CRMBot()
    bot.run(lang=ui_lang, brand=ui_brand, verbose=2)
