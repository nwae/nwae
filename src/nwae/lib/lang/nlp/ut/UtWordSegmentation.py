# -*- coding: utf-8 -*-

import nwae.config.Config as cf
import nwae.lib.lang.LangFeatures as lf
import nwae.utils.UnitTest as ut
import nwae.lib.lang.LangHelper as langhelper


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

    def do_unit_test(
            self,
            word_segmenter,
            # Sentence with expected split sentence
            list_sent_exp
    ):
        test_results = []
        for sent_exp in list_sent_exp:
            sent = sent_exp[0]
            sent_split = word_segmenter.segment_words(
                text = sent,
                return_array_of_split_words = True
            )
            test_results.append(sent_split)

        res = ut.UnitTest.get_unit_test_result(
            input_x         = [x[0] for x in list_sent_exp],
            result_test     = test_results,
            result_expected = [x[1] for x in list_sent_exp]
        )
        return res

    def get_word_segmenter(self, lang):
        return langhelper.LangHelper.get_word_segmenter(
            lang = lang,
            dirpath_wordlist     = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist     = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
            dirpath_app_wordlist = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
            postfix_app_wordlist = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
            dirpath_synonymlist  = self.config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synonymlist  = self.config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
            # During training, we don't care about allowed root words
            # We just take the first word in the synonym list as root
            # word. Only during detection, we need to do this to make
            # sure that whatever word we replace is in the feature list.
            allowed_root_words = None,
            do_profiling = False
        ).wseg

    def test_chinese(self):
        list_sent_exp = [
            # TODO Need to add '淡定' to word list
            ['中美人工智能竞赛 AI鼻祖称白宫可以更淡定',
             ['中','美','人工智能','竞赛','AI','鼻祖','称','白宫','可以','更','淡','定']],
            ['1997年，米切尔出版《机器学习》',
             ['1997','年','，','米切尔','出版','《','机器','学习','》']],
            ['米切尔（Tom Michell）教授被称为机器学习之父',
             ['米切尔','（','Tom','Michell','）','教授','被','称为','机器','学习','之','父']],
            ['美国有更多研发人工智能训练和经验积累的公司',
             ['美国','有','更','多','研发','人工智能','训练','和','经验','积累','的','公司']],
            ['一旦政府决定建立覆盖全国的医疗记录电子文档数据库…',
             ['一旦','政府','决定','建立','覆盖','全国','的','医疗','记录','电子','文档','数据库','…']],
            ['香港抗议 盘点本周最新出现的五个重大情况',
             ['香港','抗议','盘点','本周','最新','出现','的','五个','重大','情况']],
            # Other languages
            ['English Test + 中文很难 + ภาษาไทย and 한국어 ..',
             ['English','Test','+','中文','很','难','+','ภาษาไทย','and','한국어','.','.']]
        ]
        retv = self.do_unit_test(
            word_segmenter = self.get_word_segmenter(lang = lf.LangFeatures.LANG_CN),
            list_sent_exp  = list_sent_exp
        )

        return retv

    def test_thai(self):
        list_sent_exp = [
            # TODO Add 'ผิดหวัง' to dictionary
            ['บัวขาว บัญชาเมฆ ไม่ทำให้แฟนมวยชาวไทยผิดหวัง',
             ['บัว','ขาว','บัญชา','เมฆ','ไม่','ทำ','ให้','แฟน','มวย','ชาว','ไทย','ผิด','หวัง']],
            ['วันที่ 27 ต.ค. ศึก Mas Fight ที่กรุงพนมเปญ ประเทศกัมพูชา คู่เอก',
             ['วัน','ที่','27','ต','.','ค','.','ศึก','Mas','Fight','ที่','กรุง','พนม','เปญ','ประเทศ','กัมพูชา','คู่','เอก']],
            # TODO Fix this 'น็อก' should be one word, this is tricky because we look from longest to shortest
            ['ผลตัดสินแพ้ชนะอยู่ที่การน็อก หรือขอยอมแพ้เท่านั้น',
             ['ผล','ตัด','สิน','แพ้','ชนะ','อยู่','ที่','การ','น็','อก','หรือ','ขอ','ยอม','แพ้','เท่า','นั้น']],
            ['เนื่องจากสภาพแวดล้อมแห้งแล้งมาก อากาศร้อนและกระแสลมแรง',
             ['เนื่อง','จาก','สภาพ','แวด','ล้อม','แห้ง','แล้ง','มาก','อากาศ','ร้อน','และ','กระแส','ลม','แรง']],
            ['ซึ่งอยู่ห่างจากตอนเหนือของเมืองบริสเบน ประมาณ 650 กิโลเมตร,',
             ['ซึ่ง','อยู่','ห่าง','จาก','ตอน','เหนือ','ของ','เมือง','บ','ริ','ส','เบน','ประมาณ','650','กิโล','เมตร',',']],
            # Other languages
            ['English Test + 中文很难 + ภาษาไทย and 한국어 ..',
             ['English', 'Test', '+', '中文很难', '+', 'ภาษา','ไทย', 'and', '한국어', '.', '.']]
        ]
        retv = self.do_unit_test(
            word_segmenter = self.get_word_segmenter(lang = lf.LangFeatures.LANG_TH),
            list_sent_exp  = list_sent_exp
        )
        return retv

    def test_viet(self):
        list_sent_exp = [
            # TODO Split out the comma from 'trắng,'
            ['bơi cùng cá mập trắng, vảy núi lửa âm ỉ',
             ['bơi','cùng','cá mập','trắng,','vảy','núi lửa','âm ỉ']],
            ['Disney đã sản xuất một vài bộ phim đình đám vào thời điểm đó',
             ['Disney','đã','sản xuất','một vài','bộ','phim','đình đám','vào','thời điểm','đó']],
            ['nhưng Frozen là một trong những thành công đáng kinh ngạc nhất',
             ['nhưng','Frozen','là','một','trong','những','thành công','đáng','kinh ngạc','nhất']],
            # Other languages
            ['English Test + 中文很难 + ภาษาไทย and 한국어 ..',
             ['English', 'Test', '+', '中文很难', '+', 'ภาษาไทย', 'and', '한국어', '..']]
        ]
        retv = self.do_unit_test(
            word_segmenter = self.get_word_segmenter(lang = lf.LangFeatures.LANG_VN),
            list_sent_exp  = list_sent_exp
        )
        return retv

    def test_en(self):
        list_sent_exp = [
            ['async worker such as gevent/meinheld/eventlet',
             ['async','worker','such','as','gevent','/','meinheld','/','eventlet']],
            ['it doesn\'t feature the terms "capture group".',
             ['it','doesn\'t','feature','the','terms','"','capture','group','"','.']]
        ]
        return self.do_unit_test(
            word_segmenter = self.get_word_segmenter(lang = lf.LangFeatures.LANG_EN),
            list_sent_exp  = list_sent_exp
        )


if __name__ == '__main__':
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/default.cf'
    )

    tst = testNLP(
        config = config
    )
    res_cn = tst.test_chinese()
    res_th = tst.test_thai()
    res_vi = tst.test_viet()
    res_en = tst.test_en()

    total_ok = res_cn.count_ok + res_th.count_ok + res_vi.count_ok + res_en.count_ok
    total_fail = res_cn.count_fail + res_th.count_fail + res_vi.count_fail + res_en.count_fail

    print('***** RESULT *****')
    print("PASSED " + str(total_ok) + ', FAILED ' + str(total_fail))


