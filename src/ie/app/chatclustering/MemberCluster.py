#!/usr/bin/python
# -*- coding: utf-8 -*-

import ie.app.ConfigFile as cf
import ie.lib.chat.bot.Intent as lb
import ie.lib.chat.classification.ChatClustering as cc
import ie.app.CommandLine as cmdline


class MemberCluster:

    def __init__(
            self,
            lang,
            brand,
            datefrom,
            dateto,
            maxlines=0,
            chatdata_date_format = '%Y-%m-%d %H:%M:%S'
    ):
        self.lang = lang
        self.brand = brand

        self.currency = 'CNY'
        if self.lang == 'th':
            self.currency = 'THB'
        elif self.lang == 'vn':
            self.currency = 'VND'

        self.datefrom = datefrom
        self.dateto = dateto

        self.maxlines = maxlines

        self.chatdata_date_format = chatdata_date_format

        self.dirpath_synonymlist = cf.ConfigFile.DIR_SYNONYMLIST
        self.dir_rfv_commands = cf.ConfigFile.DIR_RFV_INTENTS

        return

    def extract_chat_by_first_non_greeting_member_line(self):

        # We use our LeBot to detect greetings/abuse/etc. so we know which line to start taking from member for clustering
        lebot = lb.Intent(
            lang                = self.lang,
            brand               = self.brand,
            dir_rfv_commands    = self.dir_rfv_commands,
            dirpath_synonymlist = self.dirpath_synonymlist
        )

        lebot.load_rfv_commands_from_file()
        ca = cc.ChatClustering(
            lang = self.lang,
            brand = self.brand,
            currency = self.currency,
            datefrom = self.datefrom,
            dateto = self.dateto,
            lebot = lebot,
            chatdata_date_format = self.chatdata_date_format
        )

        # Preprocess first
        ca.preprocess_chatdata_into_member_first_non_greeting_line(maxlines=self.maxlines, verbose=1)

        return

    def cluster_by_first_non_greeting_member_line(self):

        ca = cc.ChatClustering(
            lang = self.lang,
            brand = self.brand,
            currency = self.currency,
            datefrom = self.datefrom,
            dateto = self.dateto,
            lebot = None
        )

        chat_cluster = ca.analyze(no_keywords=50, verbose=1)
        print(chat_cluster[0:5])

        return

    def run(self):
        while True:
            print('Choices')
            print('1: Preprocessing: Extract from Chat Logs 1st Non-Greeting Member Line')
            print('2: Cluster Preprocessed File')
            print('e: Exit')
            user_choice = input('Enter Choice: ')

            if user_choice == '1':
                self.extract_chat_by_first_non_greeting_member_line()
            elif user_choice == '2':
                self.cluster_by_first_non_greeting_member_line()
            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')


if __name__ == '__main__':
    ui_lang = None
    ui_brand = None

    while ui_lang is None:
        ui_lang = cmdline.CommandLine.get_user_input_language()

    while ui_brand is None:
        ui_brand = cmdline.CommandLine.get_user_input_brand()

    mc = MemberCluster(
        lang=ui_lang,
        brand=ui_brand,
        datefrom='2018-07-01',
        dateto='2018-09-30',
        maxlines=100000,
        chatdata_date_format='%d/%m/%Y %H:%M'
    )
    #cc.ChatClustering.INTENT_GREETING = 'common/ยินดีต้อนรับ'
    cc.ChatClustering.INTENT_GREETING = 'common/乐天堂常见问题'
    mc.run()
