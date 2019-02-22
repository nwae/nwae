# -*- coding: utf-8 -*-

import ie.app.ConfigFile as cf
import ie.app.CommandLine as cmdline
import ie.app.chatbot.BotTrain as btrain
import ie.app.chatbot.BotTest as btest
import ie.app.chatbot.CRMBot as crmbot
import ie.app.chatclustering.MemberCluster as chatcluster
import ie.lib.chat.classification.ChatClustering as cc
import ie.app.lang.LanguageStats as ls
import ie.app.test.testNLP as tnlp
import ie.lib.lang.classification.TextClusterBasic as tc
import ie.lib.util.Log as lg
import re
import sys


class Run:

    def __init__(self):
        return

    def run(self):
        while True:
            print('Welcome to GBot Suite of Programs. Select the program you wish to run:')
            print('  Bot Related Programs:')
            print('    [BotTrain] - This is the first step in training GBot, you will be guided step by step.')
            print('    [BotTest] - Once trained, we can test it using this program. ')
            print('    [CRMBot] - Test the GBot Engine (no state) on command line.')
            print('  Information Extraction NLP Programs')
            print('    [ChatCluster] - Clusters chats by first non-greeting lines')
            print('  Just Some NLP Demos & Tests')
            print('    [LangStats] - Calculate some language stats')
            print('    [NLPTest] - It just runs some NLP demos.')
            print('    [TextClusterTest] - Some cluster demo.')
            print('')

            in_prog = input('Enter Program # or Name: ')
            if re.search(pattern='^[0-9]+$', string=in_prog):
                prog = int(in_prog)

            ui_lang = None
            ui_brand = None

            if in_prog.lower() == 'bottrain':
                while ui_lang is None:
                    ui_lang = cmdline.CommandLine.get_user_input_language()

                while ui_brand is None:
                    ui_brand = cmdline.CommandLine.get_user_input_brand()

                bt = btrain.BotTrain(lang=ui_lang, brand=ui_brand)
                bt.run()
                break

            elif in_prog.lower() == 'bottest':
                while ui_lang is None:
                    ui_lang = cmdline.CommandLine.get_user_input_language()

                while ui_brand is None:
                    ui_brand = cmdline.CommandLine.get_user_input_brand()

                bt = btest.BotTest(lang=ui_lang, brand=ui_brand)
                bt.run()
                break

            elif in_prog.lower() == 'crmbot':
                while ui_lang is None:
                    ui_lang = cmdline.CommandLine.get_user_input_language()

                while ui_brand is None:
                    ui_brand = cmdline.CommandLine.get_user_input_brand()

                bot = crmbot.CRMBot()
                bot.run(lang=ui_lang, brand=ui_brand, verbose=2)
                break

            elif in_prog.lower() == 'chatcluster':
                ui_lang = None
                ui_brand = None

                while ui_lang is None:
                    ui_lang = cmdline.CommandLine.get_user_input_language()

                while ui_brand is None:
                    ui_brand = cmdline.CommandLine.get_user_input_brand()

                mc = chatcluster.MemberCluster(
                    lang=ui_lang,
                    brand=ui_brand,
                    datefrom='2018-08-01',
                    dateto='2018-08-10',
                    maxlines=100000
                )
                cc.ChatClustering.INTENT_GREETING = 'common/ยินดีต้อนรับ'
                mc.run()
                break

            elif in_prog.lower() == 'langstats':
                while ui_lang is None:
                    ui_lang = cmdline.CommandLine.get_user_input_language()
                lstats = ls.LanguageStats(lang=ui_lang)
                lstats.run()
                break

            elif in_prog.lower() == 'nlptest':
                nlp = tnlp.testNLP()
                nlp.run_tests()
                break

            elif in_prog.lower() == 'textclustertest':
                testtc = tc.TextClusterBasic.UnitTest(verbose=1)
                testtc.run_tests()
                break

            else:
                print('Invalid choice: [' + in_prog + ']')
                print('')


if __name__ == '__main__':
    #
    # Run like '/usr/local/bin/python3.6 -m ie.app.chatbot.server.BotServer brand=fun88 lang=cn'
    #
    # Default values
    pv = {
        'topdir': None,
        'debug': '1'
    }
    args = sys.argv
    usage_msg = 'Usage: ./run.intentapi.sh topdir=/home/mark/git/mozg.nlp'

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
    lg.Log.log('Using the following parameters..')
    lg.Log.log(str(pv))

    Run().run()