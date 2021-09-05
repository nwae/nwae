#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import pandas as pd
import nwae.utils.FileUtils as futil
import nwae.utils.StringUtils as sutil
import nwae.lang.LangFeatures as lf
import nwae.lang.nlp.LatinEquivalentForm as lef
import nwae.lang.characters.LangCharacters as langchar
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.utils.UnitTest as ut


#
# Any simple word list, stopwords, etc. that can be read line by line as a single word, with no other properties
#
# Example Word Lists
#   - Chinese: i)  https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big
#              ii) https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small
#              Both above taken from the project jieba - https://github.com/fxsjy/jieba
#   - Thai:    https://github.com/pureexe/thai-wordlist
#
class WordList:

    COL_WORD = 'Word'
    COL_WORD_NUMBER = 'WordNumber'
    COL_LATIN = 'WordLatin'
    COL_LATIN_NUMBER = 'WordLatinNumber'
    # In the case of languages with syllable separator (e.g. Vietnamese, Korean),
    # the word length is actually the ngram length. In other cases, it is just the word length
    COL_NGRAM_LEN = 'WordLen'

    MAX_NGRAMS = 20

    def __init__(
            self,
            lang,
            dirpath_wordlist,
            postfix_wordlist
    ):
        self.lang = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )

        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist

        self.wordlist = None
        # Break the wordlist into ngrams for faster word segmentation
        self.ngrams = {}

        self.lang_feature = lf.LangFeatures()
        self.syl_split_token = self.lang_feature.get_split_token(
            lang  = self.lang,
            level = lf.LangFeatures.LEVEL_SYLLABLE
        )
        if self.syl_split_token is None:
            self.syl_split_token = ''
        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(lang) + '" syllable split token is "' + self.syl_split_token + '"')

        self.__load_wordlist()
        return

    def __load_wordlist(
            self
    ):
        if self.wordlist is None:
            self.wordlist = self.__load_list(
                dirpath = self.dirpath_wordlist,
                postfix = self.postfix_wordlist
            )
            self.update_ngrams()
        return

    def update_ngrams(self):
        try:
            # Get the unique length unigrams
            max_length = int( max( set(self.wordlist[WordList.COL_NGRAM_LEN]) ) )
            max_length = int( min(max_length, WordList.MAX_NGRAMS ) )

            if max_length < 1:
                errmsg =\
                    ': Lang "' + str(self.lang) + '" have no ngrams!! Max ngram length = ' + str(max_length) + '!!'
                log.Log.error(errmsg)
                raise Exception(errmsg)

            for i in range(1, max_length+1, 1):
                condition = self.wordlist[WordList.COL_NGRAM_LEN] == i
                self.ngrams[i] = self.wordlist[WordList.COL_WORD][condition].tolist()
                log.Log.debugdebug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang)
                    + '" ngrams [' + str(i) + '] (list len = ' + str(len(self.ngrams[i])) + '):\n\r'
                    + str(self.ngrams[i])
                )
        except Exception as ex:
            errmsg = \
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Exception updating ngrams list for lang "' + str(self.lang)\
                + '", exception message: ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise Exception(errmsg)

        return

    def append_wordlist(
            self,
            dirpath     = None,
            postfix     = None,
            array_words = None,
    ):
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang ' + str(self.lang) + '" Initial wordlist length = ' + str(self.wordlist.shape[0]) + '.'
            + ', appending wordlist:\n\r' + str(array_words)
        )
        wordlist_additional = None
        if array_words is not None:
            wordlist_additional = self.__load_list(
                dirpath     = None,
                postfix     = None,
                array_words = array_words
            )
        else:
            wordlist_additional = self.__load_list(
                dirpath = dirpath,
                postfix = postfix
            )
        # Join general and application wordlist
        self.wordlist = self.wordlist.append(wordlist_additional)
        # Remove duplicates
        self.wordlist = self.wordlist.drop_duplicates(subset=[WordList.COL_WORD])
        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Lang "' + str(self.lang) + '" final wordlist length = ' + str(self.wordlist.shape[0]) + '.'
        )

        self.update_ngrams()

        return

    # General function to load wordlist or stopwords
    def __load_list(
            self,
            dirpath,
            postfix,
            array_words = None
    ):

        lc = langchar.LangCharacters()

        content = None
        if array_words is not None:
            content = array_words
        else:
            filepath = str(dirpath) + '/' + str(self.lang) + str(postfix)
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" loading list for [' + self.lang + ']' + '[' + filepath + ']'
            )

            fu = futil.FileUtils()
            content = fu.read_text_file(filepath)

            #
            # We will not tolerate missing file. This is because when adding user wordlists,
            # it will significantly effect Bot quality.
            # For example if file is missing, we will miss out on user keywords like "必威" or
            # "云闪付" or "彩金", etc, which will severely reduce Bot efficiency.
            #
            if len(content) == 0:
                warning_msg =\
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                    + ': Lang "' + str(self.lang) + '" file [' + filepath + '] is empty or non-existent!!'
                log.Log.warning(warning_msg)

            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Lang "' + str(self.lang) + '" read from file "' + str(filepath)
                + '" ' + str(len(content)) + ' lines.'
            )

        words = []
        # Convert words to some number
        measures = []
        # In Latin form
        words_latin = []
        measures_latin = []
        #
        # TODO Don't loop
        #
        for line in content:
            line = str(line)
            line = sutil.StringUtils.trim(line)
            # Remove empty lines
            if len(line)<=0: continue
            # Remove comment lines starting with '#'
            if re.match(u'^#', line): continue

            word = line

            # Make sure to convert all to Unicode
            # word = unicode(word, encoding='utf-8')
            # Remove empty words
            if len(word)<=0: continue

            words.append(word)
            measures.append(lc.convert_string_to_number(word))

            wordlatin = lef.LatinEquivalentForm.get_latin_equivalent_form(lang=self.lang, word=word)
            words_latin.append(wordlatin)
            measures_latin.append(lc.convert_string_to_number(wordlatin))

        try:
            # Convert to pandas data frame
            df_wordlist = pd.DataFrame({
                WordList.COL_WORD        : words,
                WordList.COL_WORD_NUMBER : measures,
                WordList.COL_LATIN       : words_latin,
                WordList.COL_LATIN_NUMBER: measures_latin,
                WordList.COL_NGRAM_LEN   : [0] * len(words)
            })
            if self.syl_split_token == '':
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '" ngram length for ' + self.lang + ' is just the WORD length.'
                )
                if df_wordlist.shape[0] > 0:
                    df_wordlist[WordList.COL_NGRAM_LEN] = pd.Series(data=words).str.len()
            else:
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Lang "' + str(self.lang) + '" ngram length for ' + self.lang + ' is just the SYLLABLE length.'
                )
                if df_wordlist.shape[0] > 0:
                    # TODO FutureWarning: The default value of regex will change from True to False in a future version.
                    # We remove all non-space character, so we only count the spaces+1, which is the unigram length
                    df_wordlist[WordList.COL_NGRAM_LEN] = \
                        pd.Series(data=words).str.replace('[^ ]', '', regex=True).str.len() + 1

            df_wordlist = df_wordlist.drop_duplicates(subset=[WordList.COL_WORD])
            # Need to reset indexes, otherwise some index will be missing
            df_wordlist = df_wordlist.reset_index(drop=True)

            return df_wordlist
        except Exception as ex:
            errmsg =\
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                + ': Lang ' + str(self.lang) + ' wordlist loading exception: ' + str(ex) + '.'
            log.Log.error(errmsg)
            raise Exception(errmsg)


class WordlistUnitTest:

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        # Test mainly for languages that require word lists when doing word segmentation
        test_lang_wl = {
            lf.LangFeatures.LANG_ZH: {
                1: ('出', '或', '日', '由', '里', '用', '所', '向', '已', '其', '给', '很', '看', '使', '前',
                    '新', '想', '却', '它', '云', '层'),
                2: ('国家', '可以', '发展', '这个', '工作', '这样', '全国', '经济', '这些', '不是', '社会', '记者',
                    '面积', '人员'),
                3: ('委员会', '武汉市', '国民党', '韦小宝', '为什么', '实际上'),
                4: ('人民政府', '社会主义', '人民法院'),
                5: ('天安门广场', '中国共产党', '中国共产党'),
                6: ('人民代表大会'),
                7: ('全国人大常委会')
            },
            lf.LangFeatures.LANG_TH: {
                2: ('กง', 'มา', 'ทำ', 'ชน', 'ใน', 'จะ'),
                3: ('ผู้', 'ราย', 'ว่า', 'งาน', 'ดึก', 'วัน', 'ที่', 'ไทย', 'ถึง', 'ชี้','แจง', 'มวล', 'การ', 'แจก',
                    'ของ', 'ตาม', 'ได้', 'ให้', 'กับ', 'ทุก', 'โดย'),
                4: ('สื่อ', 'ข่าว', 'ช่วง', 'กลาง', 'ผ่าน', 'กรณี', 'หรือ', 'หน้า', 'สถาน', 'ทั่ว', 'นั้น'),
                5: ('การณ์', 'จำนวน'),
                6: ('ดำเนิน', 'อนามัย', 'ประเทศ', 'บริหาร'),
                7: ('กระทรวง', 'มหาดไทย', 'หนังสือ', 'จังหวัด'),
                8: ('กระเทือน', 'โก้งเก้ง'),
                9: ('จ้องหน่อง', 'ดาวประดับ')
            },
            lf.LangFeatures.LANG_VI: {
                1: ('a-đa', 'A-đam', 'A-đi-xơn', 'a-đrê-na-lin', 'a-ga', 'a-giăng', 'a-giăng-đa', 'a-gon',
                    'a-vô-ca', 'a-xen', 'a-xê-ti-len', 'a-xê-ton', 'a-xê-tôn','ampe', 'ampere', 'ampli',
                    'ăm-bờ-ra-da', 'ăm-pe', 'ăm-pun', 'ẵm', 'ăn', 'ăng-ga', 'ăng-ten', 'ăng-ti-gôn', 'ăng-ti-moan',
                    'ăng-tơ-ra-xit', 'mương', 'mường', 'Mường', 'mướp', 'mướt', 'mượt', 'mưỡu', 'mứt', 'mưu', 'mỹ',
                    'xô-viết', 'xồ', 'xổ', 'xốc', 'xộc', 'xôđa', 'xôi', 'xổi', 'xối', 'xôm'),
                2: ('cận lai', 'cận lao', 'cận lân', 'cận lợi', 'cận nhật', 'cận răng', 'cận sản', 'cận sử',
                    'cận tâm','câu giam', 'câu giăng', 'Câu Gồ', 'câu hát', 'câu hoạ', 'câu hỏi', 'câu kéo',
                    'câu kẹo', 'câu kết', 'câu khách', 'mằn thắn', 'mắn đẻ', 'mặn mà', 'mặn miệng', 'mặn mòi',
                    'mặn nồng', 'mặn tình', 'thu tiếng', 'thu tóm', 'thu tô', 'thu vén', 'thu xếp', 'thù ân',
                    'thù du', 'Xuân Nội', 'Xuân Nộn', 'xuân nữ', 'xuân phân', 'xuân phong', 'Xuân Phong'),
                3: ('điều kiện cần', 'điều kiện đủ', 'điều phối viên', 'điệu này (thì)', 'đinh đóng cột',
                    'tương lai học', 'tường cánh gà', 'tượng trưng hóa', 'tửu tinh kế', 'tỷ lệ thức',
                    'vật hậu học', 'vật lí học', 'vật linh giáo', 'vật thể hóa', 'vật tổ giáo', 'vật tự nó'),
                4: ('bẻ cành cung quế', 'bẻ đũa cả nắm', 'bẻ gãy sừng trâu', 'bẻ hành bẻ tỏi', 'bẻ nạng chống trời',
                    'đạo thầy nghĩa tớ', 'đạo vợ nghĩa chồng', 'đạp sỏi giày sành', 'đạp tuyết tầm mai',
                    'mò kim rốn bể', 'móc mắt lôi mề', 'mọc lông mọc cánh', 'mọc lông trong bụng', 'mọc mũi sủi tăm',
                    'rau nào sâu ấy', 'rau súng ăn gỏi', 'rày gió mai mưa', 'rày nắng mai mưa', 'rày ước mai ao'),
                5: ('ăn như Nam-hạ vác đất', 'ăn như tằm ăn rỗi', 'ăn quà như mỏ khoét', 'ăn rồi lại nằm mèo',
                    'ăn thủng nồi trôi rế','có dại mới nên khôn', 'có danh không có thực', 'có đẻ không có nuôi',
                    'kẻ nhát nát người bạo', 'kẻ nửa cân, người tám', 'kẻ tám lạng người nửa', 'kéo cày trả (giả) nợ',
                    'xấu mã có duyên thầm', 'xây lâu đài trên cát', 'xe chỉ buộc chân voi', 'xé mắm hòng mút tay'),
            },
        }

        dirpath_wl = self.ut_params.dirpath_wordlist
        postfix_wl = self.ut_params.postfix_wordlist

        for lang in test_lang_wl.keys():
            # print('Unit Test lang ' + str(lang))
            ngrams_test = test_lang_wl[lang]
            wl = WordList(
                lang             = lang,
                dirpath_wordlist = dirpath_wl,
                postfix_wordlist = postfix_wl
            )

            for len in ngrams_test.keys():
                words = ngrams_test[len]
                # For tuple with a single item, Python reduces it to the item type
                if type(words) not in [tuple, list]:
                    words = [words]
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Test words ' + str(words) + ' in word list ' + str(wl.ngrams[len])
                )
                for w in words:
                    is_w_in_list = w in wl.ngrams[len]
                    # print('Test Word "' + str(w) + '" in = ' + str(is_w_in_list))
                    res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                        observed = is_w_in_list,
                        expected = True,
                        test_comment = 'Test "' + str(w) + '" in ' + str(len) + '-gram'
                    ))

        return res_final



if __name__ == '__main__':
    import nwae.config.Config as cf

    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_PATH_DEFAULT
    )
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_INFO
    log.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    res = WordlistUnitTest(ut_params=ut_params).run_unit_test()
    print('PASSED ' + str(res.count_ok) + ', FAILED ' + str(res.count_fail))
    exit(0)

    for lang in ['vn']:
        dirpath_wl = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST)
        postfix_wl = config.get_config(cf.Config.PARAM_NLP_POSTFIX_WORDLIST)
        print('Dir "' + str(dirpath_wl) + '", postfix "' + str(postfix_wl) + '"')
        wl = WordList(
            lang             = lang,
            dirpath_wordlist = dirpath_wl,
            postfix_wordlist = postfix_wl
        )
        log.Log.log('')
        log.Log.log( lang + ': Read Word List ' + str(wl.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = wl.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df[WordList.COL_WORD].loc[i]) + ','
            sm = sm + str(df[WordList.COL_WORD_NUMBER].loc[i]) + ','
            s_latin = s_latin + str(df[WordList.COL_LATIN].loc[i]) + ','
            sm_latin = sm_latin + str(df[WordList.COL_LATIN_NUMBER].loc[i]) + ','

        log.Log.log ( s )
        log.Log.log ( sm )
        log.Log.log ( s_latin )
        log.Log.log ( sm_latin )

        # Stopwords
        dirpath_wl = config.get_config(cf.Config.PARAM_NLP_DIR_WORDLIST)
        postfix_wl = config.get_config(cf.Config.PARAM_NLP_POSTFIX_STOPWORDS)
        print('Dir "' + str(dirpath_wl) + '", postfix "' + str(postfix_wl) + '"')
        sw = WordList(
            lang             = lang,
            dirpath_wordlist = dirpath_wl,
            postfix_wordlist = postfix_wl
        )
        log.Log.log('')
        log.Log.log ( lang + ': Read Stopword List ' + str(sw.wordlist.shape[0]) + " lines" )
        s = ''
        sm = ''
        s_latin = ''
        sm_latin = ''
        df = sw.wordlist
        for i in range(0, min(100, df.shape[0]), 1):
            s = s + str(df[WordList.COL_WORD].loc[i]) + ','
            sm = sm + str(df[WordList.COL_WORD_NUMBER].loc[i]) + ','
            s_latin = s_latin + str(df[WordList.COL_LATIN].loc[i]) + ','
            sm_latin = sm_latin + str(df[WordList.COL_LATIN_NUMBER].loc[i]) + ','

        log.Log.log ( s )
        log.Log.log ( sm )
        log.Log.log ( s_latin )
        log.Log.log ( sm_latin )

