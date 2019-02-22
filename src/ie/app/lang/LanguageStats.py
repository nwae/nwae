#!/usr/bin/python
# -*- coding: utf-8 -*-

import ie.app.ConfigFile as cf
import ie.lib.lang.stats.LangStats as ls


#
# Here we calculate
#   1. Basic token (character, syllable or word) frequency stats in languages
#   2.
#
class LanguageStats:

    def __init__(self, lang):
        self.lang = lang
        self.lang_stats = ls.LangStats(dirpath_traindata   = cf.ConfigFile.DIR_NLP_LANGUAGE_TRAINDATA,
                                       dirpath_collocation = cf.ConfigFile.DIR_NLP_LANGUAGE_STATS_COLLOCATION)
        return

    def calculate_language_unigram_distribution_stats(self):
        df = self.lang_stats.get_lang_unigram_distribution_stats(lang=self.lang, verbose=1)

        print('Read ' + str(len(df['Lines'])) + ' lines from training files.')
        print('Read ' + str(len(df['TokenList'])) + ' characters from training files.')
        df_freqtable = df['FreqTable']
        print(df_freqtable[0:100])
        return

    def calculate_character_sequence_probability(self):
        df = self.lang_stats.get_character_sticky_distribution_stats(lang=self.lang, verbose=1)
        print(df)
        df.to_csv(self.lang_stats.dirpath_collocation + '/collocation.stats.cn.csv', ',')

        self.lang_stats.load_collocation_stats()
        # print ( LangStats.collocation_stats )

        prechar = '天'
        postchar = '堂'
        # Probability of 'prechar' given 'postchar' is the post character
        print(self.lang_stats.get_collocation_probability('cn', prechar, postchar, 'post', 1))
        # Probability of 'postchar' given 'prechar' is the pre character
        print(self.lang_stats.get_collocation_probability('cn', prechar, postchar, 'pre', 1))
        return

    def run(self):
        while True:
            print('Choices (language=' + self.lang + ')')
            print('1: Calculate Unigram Distribution Stats')
            print('2: Calculate Token Sequence Probability')
            print('e: Exit')
            user_choice = input('Enter Choice: ')

            if user_choice == '1':
                self.calculate_language_unigram_distribution_stats()
            elif user_choice == '2':
                self.calculate_character_sequence_probability()
            elif user_choice == 'e':
                break
            else:
                print('No such choice [' + user_choice + ']')

