# -*- coding: utf-8 -*-

import nwae.ConfigFile as cf
import nwae.lib.lang.nlp.WordSegmentation as ws
import nwae.utils.Log as lg


#
# Test NLP stuff
#
class testNLP:

    def __init__(
            self,
            config
    ):
        self.config = config
        return

    def test_word_segmentation(self):
        self.test_chinese()
        self.test_thai()
        self.test_viet()
        self.test_ml()
        return

    def test_chinese(self):
        ws_cn = ws.WordSegmentation(
            lang             = 'cn',
            dirpath_wordlist = self.config.DIR_WORDLIST,
            postfix_wordlist = self.config.POSTFIX_WORDLIST,
            do_profiling     = self.config.DO_PROFILING
        )
        # Add application wordlist
        ws_cn.add_wordlist(
            dirpath = self.config.DIR_APP_WORDLIST,
            postfix = self.config.POSTFIX_APP_WORDLIST
        )

        # Simplified Chinese
        # TODO: '多乐币' is split wrongly.
        text = '谷歌和脸书成了冤大头？我有多乐币 hello world 两间公司合共被骗一亿美元克里斯。happy当只剩两名玩家时，无论是第几轮都可以比牌。'
        print('"' + ws_cn.segment_words(text=text) + '"')
        # Test against jieba
        # text_segmented = jieba.cut(text, cut_all=False)
        # print( ' '.join(text_segmented) )

        # Traditional Chinese
        text = '榖歌和臉書成瞭冤大頭？我有多樂幣 hello world 兩間公司閤共被騙一億美元剋裏斯。happy當隻剩兩名玩傢時，無論是第幾輪都可以比牌。'
        print('Converting [' + text + '] to Simplified Chinese..')
        text_sim = ws_cn.convert_to_simplified_chinese(text)
        print(text_sim)
        print('"' + ws_cn.segment_words(text=text_sim) + '"')

        text = ' Cooper，test'
        print('"' + ws_cn.segment_words(text=text) + '"')

        return 0

    def test_thai(self):
        ws_th = ws.WordSegmentation(
            lang             = 'th',
            dirpath_wordlist = self.config.DIR_WORDLIST,
            postfix_wordlist = self.config.POSTFIX_WORDLIST,
            do_profiling     = self.config.DO_PROFILING
        )
        # Add application wordlist
        ws_th.add_wordlist(
            dirpath = self.config.DIR_APP_WORDLIST,
            postfix = self.config.POSTFIX_APP_WORDLIST,
        )

        text = 'งานนี้เมื่อต้องขึ้นแท่นเป็นผู้บริหาร แหวนแหวน จึงมุมานะไปเรียนต่อเรื่องธุ'
        print('"' + ws_th.segment_words(text=text) + '"')

        # This checks that 'รอการ' is correctly split into 'รอ การ' and not 'รอก า', using the language rule.
        text = 'ผมแจ้งฝากไปสำเร็จ. รอการดำเนินการอยู่ขณะนี้อยากทราบว่าที่ล่าช้าเป็นเพราะผมฝากผ่านตู้เงินสดไทยพณิชหรือเปล่า.'
        print('"' + ws_th.segment_words(text=text) + '"')

        # This checks that sequences of single alphabets are joined together
        text = 'ผจ้งฝากปสเร็รดำเ่า.'
        print('"' + ws_th.segment_words(text=text) + '"')
        text = 'จำ ยุ ส เซ อร์  ไม่ ได้'
        print('"' + ws_th.segment_words(text=text) + '"')

        return

    def test_viet(self):
        ws_vn = ws.WordSegmentation(
            lang             = 'vn',
            dirpath_wordlist = self.config.DIR_WORDLIST,
            postfix_wordlist = self.config.POSTFIX_WORDLIST,
            do_profiling     = self.config.DO_PROFILING
        )

        text = 'Hoặc nếu ông Thăng không bị kỷ luật, cây bút Tâm Chánh nêu giả thiết Tổng Bí thư Nguyễn Phú Trọng sẽ có thể "huy động sự tham gia của người dân vào cuộc đấu tranh sinh tử này".'.lower()
        print('"' + ws_vn.segment_words(text=text) + '"')
        return

    def test_ml(self):
        ws_cn = ws.WordSegmentation(
            lang             = 'cn',
            dirpath_wordlist = self.config.DIR_WORDLIST,
            postfix_wordlist = self.config.POSTFIX_WORDLIST,
            do_profiling     = self.config.DO_PROFILING
        )
        # Add application wordlist
        ws_cn.add_wordlist(
            dirpath = self.config.DIR_APP_WORDLIST,
            postfix = self.config.POSTFIX_APP_WORDLIST,
        )

        print(
            ws_cn.get_possible_word_separators_from_start(
                text_array = "谷歌和脸书成了冤大头？我有多乐币", max_lookforward_chars=0
            )
        )
        print(
            ws_cn.get_possible_word_separators_from_start(
                text_array = "冤大头？我有多乐币", max_lookforward_chars=0
            )
        )

        return


if __name__ == '__main__':
    config = cf.ConfigFile.get_cmdline_params_and_init_config_singleton()

    tst = testNLP(
        config = config
    )
    #lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_2
    tst.test_chinese()
    tst.test_thai()
    tst.test_viet()
    tst.test_ml()


