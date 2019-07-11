#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import sys
import ie.app.ConfigFile as cf
import ie.lib.lang.stats.LangStats as ls
import ie.lib.lang.nlp.WordSegmentation as ws

#
# Test NLP stuff
#
class testNLP:

    def __init__(self, verbose=0):
        self.lang_stats = ls.LangStats(
            dirpath_traindata   = cf.ConfigFile.DIR_NLP_LANGUAGE_TRAINDATA,
            dirpath_collocation = cf.ConfigFile.DIR_NLP_LANGUAGE_STATS_COLLOCATION
        )
        self.lang_stats.load_collocation_stats()
        self.verbose = verbose
        return

    def test_word_segmentation(self):
        self.test_chinese()
        self.test_other()
        self.test_ml()
        return

    def test_chinese(self):
        ws_cn = ws.WordSegmentation(
            lang             = 'cn',
            dirpath_wordlist = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            lang_stats       = self.lang_stats,
            do_profiling     = True,
            verbose          = self.verbose
        )
        # Add application wordlist
        ws_cn.add_wordlist(
            dirpath = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix = '.' + 'cn.fun88' + cf.ConfigFile.POSTFIX_APP_WORDLIST,
        )

        # Simplified Chinese
        # TODO: '多乐币' is split wrongly.
        text = '谷歌和脸书成了冤大头？我有多乐币 hello world 两间公司合共被骗一亿美元克里斯。happy当只剩两名玩家时，无论是第几轮都可以比牌。'
        print(ws_cn.segment_words(text=text))
        # Test against jieba
        # text_segmented = jieba.cut(text, cut_all=False)
        # print( ' '.join(text_segmented) )

        # Traditional Chinese
        text = '榖歌和臉書成瞭冤大頭？我有多樂幣 hello world 兩間公司閤共被騙一億美元剋裏斯。happy當隻剩兩名玩傢時，無論是第幾輪都可以比牌。'
        print('Converting [' + text + '] to Simplified Chinese..')
        text_sim = ws_cn.convert_to_simplified_chinese(text)
        print(text_sim)
        print(ws_cn.segment_words(text=text_sim))

        text = ' Cooper，test'
        print(ws_cn.segment_words(text=text))

        return 0

    def test_other(self):
        ws_th = ws.WordSegmentation(
            lang             = 'th',
            dirpath_wordlist = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            lang_stats       = self.lang_stats,
            do_profiling     = True,
            verbose          = self.verbose
        )
        # Add application wordlist
        ws_th.add_wordlist(
            dirpath = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix = '.' + 'th.fun88' + cf.ConfigFile.POSTFIX_APP_WORDLIST,
        )

        ws_vn = ws.WordSegmentation(
            lang             = 'vn',
            dirpath_wordlist = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            lang_stats       = self.lang_stats,
            do_profiling     = True,
            verbose          = self.verbose
        )

        text = 'งานนี้เมื่อต้องขึ้นแท่นเป็นผู้บริหาร แหวนแหวน จึงมุมานะไปเรียนต่อเรื่องธุ'
        print(ws_th.segment_words(text=text, join_single_alphabets=True))

        # This checks that 'รอการ' is correctly split into 'รอ การ' and not 'รอก า', using the language rule.
        text = 'ผมแจ้งฝากไปสำเร็จ. รอการดำเนินการอยู่ขณะนี้อยากทราบว่าที่ล่าช้าเป็นเพราะผมฝากผ่านตู้เงินสดไทยพณิชหรือเปล่า.'
        print(ws_th.segment_words(text=text, join_single_alphabets=True))

        # This checks that sequences of single alphabets are joined together
        text = 'ผจ้งฝากปสเร็รดำเ่า.'
        print(ws_th.segment_words(text=text, join_single_alphabets=False))
        text = 'ผจ้งฝากปสเร็รดำเ่า.'
        print(ws_th.segment_words(text=text, join_single_alphabets=True))
        text = 'จำ ยุ ส เซ อร์  ไม่ ได้'
        print(ws_th.segment_words(text=text, join_single_alphabets=True))

        text = 'Hoặc nếu ông Thăng không bị kỷ luật, cây bút Tâm Chánh nêu giả thiết Tổng Bí thư Nguyễn Phú Trọng sẽ có thể "huy động sự tham gia của người dân vào cuộc đấu tranh sinh tử này".'.lower()
        print(ws_vn.segment_words(text=text, join_single_alphabets=True))

        return 0

    def test_ml(self):
        ws_cn = ws.WordSegmentation(
            lang             = 'cn',
            dirpath_wordlist = cf.ConfigFile.DIR_WORDLIST,
            postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST,
            lang_stats       = self.lang_stats,
            do_profiling     = False,
            verbose          = self.verbose
        )
        # Add application wordlist
        ws_cn.add_wordlist(
            dirpath = cf.ConfigFile.DIR_APP_WORDLIST,
            postfix = '.' + 'cn.fun88' + cf.ConfigFile.POSTFIX_APP_WORDLIST,
        )

        print(
            ws_cn.get_possible_word_separators_from_start(
                text="谷歌和脸书成了冤大头？我有多乐币", max_lookforward_chars=0
            )
        )
        print(
            ws_cn.get_possible_word_separators_from_start(
                text="冤大头？我有多乐币", max_lookforward_chars=0
            )
        )

        return


if __name__ == '__main__':
    # Default values
    pv = {
        'topdir': None,
    }
    args = sys.argv
    usage_msg = 'Usage: ./run.bottest.sh topdir=/Users/mark.tan/git/mozg.nlp'

    for arg in args:
        arg_split = arg.split('=')
        if len(arg_split) == 2:
            param = arg_split[0].lower()
            value = arg_split[1]
            if param in list(pv.keys()):
                print('Command line param [' + param + '], value [' + str(value) + '].')
                pv[param] = value

    if (pv['topdir'] is None):
        errmsg = usage_msg
        raise (Exception(errmsg))

    #
    # !!!MOST IMPORTANT, top directory, otherwise all other config/NLP/training/etc. files we won't be able to find
    #
    cf.ConfigFile.TOP_DIR = pv['topdir']
    cf.ConfigFile.top_dir_changed()

    tst = testNLP(verbose=0)
    tst.test_word_segmentation()


