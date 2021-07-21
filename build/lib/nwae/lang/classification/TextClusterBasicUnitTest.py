# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import nwae.lang.LangFeatures as lf
import nwae.utils.Log as log
from nwae.lang.classification.TextClusterBasic import TextClusterBasic
from inspect import currentframe, getframeinfo
from nwae.lang.config.Config import Config
from nwae.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
import nwae.utils.UnitTest as ut


#
# Unit Test for TextClusterBasic Class
#
class TextClusterBasicUnitTest:
    ######################################
    # Test Functions below
    #   One observation is that the TF measure is quite bad, and using our normalized form is much better.

    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def do_clustering(
            self,
            text,
            ncenters,
            # In the form [(0,1,2,4), (3,5,6)]
            # means, indexes 0, 1, 2, 4 should belong to one cluster, and indexes
            # 3, 5, 6 should belong to another
            expected_clusters,
            # Expected elements must be greater than this proportion
            test_threshold_inside = 0.5,
            # Outside elements must be less than this proportion
            test_threshold_outside = 0.5,
            test_description = '',
            feature_presence_only = False,
            freq_measure = 'tf',
            weigh_idf    = False
    ):
        res_clustering = ut.ResultObj(count_ok=0, count_fail=0)

        tc = TextClusterBasic(text)
        tc.calculate_top_keywords(remove_quartile=50)

        observed_cluster_labels = tc.cluster_text(
            ncenters      = ncenters,
            iterations    = 50,
            freq_measure  = freq_measure,
            weigh_idf     = weigh_idf,
            feature_presence_only = feature_presence_only
        )
        observed_cluster_labels = observed_cluster_labels[0]
        # Group into clusters
        observed_cluster_groups_dict = {}
        for item_index in range(observed_cluster_labels.shape[0]):
            item_lbl = observed_cluster_labels[item_index]
            if item_lbl not in observed_cluster_groups_dict.keys():
                observed_cluster_groups_dict[item_lbl] = []
            observed_cluster_groups_dict[item_lbl].append(item_index)

        observed_cluster_groups = list(observed_cluster_groups_dict.values())

        #
        # First the number of clusters must be the same
        #
        res_clustering.update_bool(
            res_bool = ut.UnitTest.assert_true(
                observed = len(observed_cluster_groups),
                expected = len(expected_clusters),
                test_comment = str(test_description) + '. Test number of clusters '
                               + str(len(observed_cluster_groups))
            )
        )

        #
        # In every observed cluster, the expected elements inside it must be >50%,
        # and outside elements < 50%
        #
        for i in range(len(expected_clusters)):
            expected_clstr = expected_clusters[i]
            observed_clstr = observed_cluster_groups[i]
            exp_inside = set(expected_clstr).intersection(observed_clstr)
            foreign_outside = set(observed_clstr).difference(expected_clstr)
            inside_proportion = round(len(exp_inside) / len(expected_clstr), 2)
            outside_proportion = round(len(foreign_outside) / len(expected_clstr), 2)
            res_clustering.update_bool(
                res_bool = ut.UnitTest.assert_true(
                    observed = inside_proportion > test_threshold_inside,
                    expected = True,
                    test_comment = str(test_description)
                                   + '. Cluster ' + str(i) + ' Inside proportion ' + str(inside_proportion)
                                   + ' > ' + str(test_threshold_inside)
                )
            )
            res_clustering.update_bool(
                res_bool = ut.UnitTest.assert_true(
                    observed = outside_proportion < test_threshold_outside,
                    expected = True,
                    test_comment = str(test_description)
                                   + '. Cluster ' + str(i) + ' Outside proportion ' + str(outside_proportion)
                                   + ' > ' + str(test_threshold_outside)
                )
            )
            log.Log.debug("Inside intersection: " + str(exp_inside) + ', outside = ' + str(foreign_outside))
            log.Log.debug('Inside proportion = ' + str(inside_proportion) + ', outside = ' + str(outside_proportion))

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Cluster labels: ' + str(observed_cluster_labels)
            + ', Cluster groups: ' + str(observed_cluster_groups)
            + ', Expected groups: ' + str(expected_clusters)
        )

        return res_clustering

    def test_textcluster_english(self):
        res = ut.ResultObj(count_ok=0, count_fail=0)

        lang = lf.LangFeatures.LANG_EN
        stopwords = [
            'the', 'of', 'in', 'on', 'and', 'or', 'to', 'be', 'a', 'is', 'are', 'at', 'as', 'for', 'this', 'that',
            'was', 'were', 'which', 'when', 'where', 'will', 'would', 'with', 'his', 'her', 'it', 'from', 'than',
            'who', 'while', 'they', 'could', 'these', 'those', 'has', 'have', 'through', 'some', 'other', 'way'
        ]
        self.txt_preprocessor = TxtPreprocessor(
            identifier_string      = str(lang) + ' test',
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = lang,
            dirpath_synonymlist    = self.ut_params.dirpath_synonymlist,
            postfix_synonymlist    = self.ut_params.postfix_synonymlist,
            dir_wordlist           = self.ut_params.dirpath_wordlist,
            postfix_wordlist       = self.ut_params.postfix_wordlist,
            dir_wordlist_app       = self.ut_params.dirpath_app_wordlist,
            postfix_wordlist_app   = self.ut_params.postfix_app_wordlist,
            stopwords_list         = stopwords,
            do_spelling_correction = False,
            do_word_stemming       = lf.LangFeatures().have_verb_conjugation(lang=lang),
            do_profiling           = False
        )

        #
        # We take a few news articles and try to automatically classify sentences belonging to the same news article.
        # This example demonstrates the need for root word extraction, which will increase accuracy significantly.
        #
        text = [
            # Article 1
            'Freezing temperatures have gripped the nation, making Wednesday the coldest day yet this winter.',
            'Morning lows plunged to minus 16-point-three degrees Celsius in Seoul , the lowest to be posted during this year’s cold season.',
            'As of 7 a.m. Wednesday , morning lows stood at minus 15-point-four degrees in Daejeon , nearly minus 22 degrees in the Daegwallyeong mountain pass in Pyeongchang and minus 14 degrees in Gangneung.',
            'Due to the wind chill factor, temperatures stood at nearly minus 23 degrees in Seoul , minus 25 in Incheon and roughly minus 36 degrees in Daegwallyeong .',
            'An official of the Korea Meteorological Administration said the nation will continue to see subzero temperatures for the time being with the central regions and some southern inland areas projected to see morning lows plunge below minus 15 degrees',
            'Currently , a cold wave warning is in place for Seoul , Incheon , Daejeon and Sejong as well as the provinces of Gangwon , Chungcheong , North Jeolla and North Gyeongsang.',
            # Article 2
            'There are two primary motivations for keeping Bitcoin''s inventor keeping his or her or their identity secret.',
            'One is privacy. As Bitcoin has gained in popularity – becoming something of a worldwide phenomenon – Satoshi Nakamoto would likely garner a lot of attention from the media and from governments.',
            'The other reason is safety. Looking at 2009 alone , 32,489 blocks were mined; at the then-reward rate of 50 BTC per block, the total payout in 2009 was 1,624,500 BTC, which at today’s prices is over $900 million.',
            'One may conclude that only Satoshi and perhaps a few other people were mining through 2009, and that they possess a majority of that $900 million worth of BTC.',
            'Someone in possession of that much BTC could become a target of criminals, especially since bitcoins are less like stocks and more like cash, where the private keys needed to authorize spending could be printed out and literally kept under a mattress.',
            'While it''s likely the inventor of Bitcoin would take precautions to make any extortion-induced transfers traceable, remaining anonymous is a good way for Satoshi to limit exposure.',
            # Article 3
            'Some of these models of concurrency are primarily intended to support reasoning and specification, while others can be used through the entire development cycle, including design, implementation, proof, testing and simulation of concurrent systems',
            'The proliferation of different models of concurrency has motivated some researchers to develop ways to unify these different theoretical models.',
            'The Concurrency Representation Theorem in the actor model provides a fairly general way to represent concurrent systems that are closed in the sense that they do not receive communications from outside.'
        ]

        text_sentences_arr = [self.txt_preprocessor.process_text(inputtext=x) for x in text]
        log.Log.debugdebug(
            'PRE-PROCESSED ' + str(lang) + ' SENTENCES:\n\r' + str(text_sentences_arr)
        )

        # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
        # do_clustering(text=text, stopwords=stopwords, ncenters=3, freq_measure='tf', weigh_idf=False, verbose=0)
        res_cluster = self.do_clustering(
            text                   = text_sentences_arr,
            ncenters               = 3,
            expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9),(10,11,12)),
            test_threshold_inside  = 0.3,
            test_threshold_outside = 0.7,
            test_description       = '1. ' + str(lang) + ' normalized, no IDF',
            freq_measure           = 'normalized',
            weigh_idf              = False
        )
        res.update(other_res_obj=res_cluster)
        # do_clustering(text=text, stopwords=stopwords, ncenters=3, freq_measure='frequency', weigh_idf=False, verbose=0)

        # Now weigh IDF
        log.Log.debugdebug('Weighing by IDF..')
        res_cluster = self.do_clustering(
            text                   = text_sentences_arr,
            ncenters               = 3,
            expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9),(10,11,12)),
            test_threshold_inside  = 0.3,
            test_threshold_outside = 0.7,
            test_description       = '2. ' + str(lang) + ' normalized, with IDF',
            freq_measure           = 'normalized',
            weigh_idf              = True
        )
        res.update(other_res_obj=res_cluster)

        # log.Log.debugdebug('Now using only feature presence (means freq is 0 or 1 only)')
        #
        # Word presence only
        #
        # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
        # do_clustering(text=text, stopwords=stopwords, ncenters=3, feature_presence_only=True, freq_measure='tf', weigh_idf=False, verbose=0)
        # res_cluster = self.do_clustering(
        #     text                   = text_sentences_arr,
        #     ncenters               = 3,
        #     expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9),(10,11,12)),
        #     test_threshold_inside  = 0.3,
        #     test_threshold_outside = 0.7,
        #     test_description       = '3. ' + str(lang) + ' normalized, no IDF, feature presence only',
        #     feature_presence_only  = True,
        #     freq_measure           = 'normalized',
        #     weigh_idf              = False
        # )
        # res.update(other_res_obj=res_cluster)

        # # Now weigh IDF
        # log.Log.debugdebug('Weighing by IDF..')
        # res_cluster = self.do_clustering(
        #     text                   = text_sentences_arr,
        #     ncenters               = 3,
        #     expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9),(10,11,12)),
        #     test_threshold_inside  = 0.3,
        #     test_threshold_outside = 0.7,
        #     test_description       = '4. ' + str(lang) + ' normalized, with IDF, feature presence only',
        #     feature_presence_only  = True,
        #     freq_measure           = 'normalized',
        #     weigh_idf              = True
        # )
        # res.update(other_res_obj=res_cluster)

        return res

    def test_textcluster_chinese(self):
        res = ut.ResultObj(count_ok=0, count_fail=0)

        lang = lf.LangFeatures.LANG_ZH
        self.txt_preprocessor = TxtPreprocessor(
            identifier_string      = str(lang) + ' test',
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = lang,
            dirpath_synonymlist    = self.ut_params.dirpath_synonymlist,
            postfix_synonymlist    = self.ut_params.postfix_synonymlist,
            dir_wordlist           = self.ut_params.dirpath_wordlist,
            postfix_wordlist       = self.ut_params.postfix_wordlist,
            dir_wordlist_app       = self.ut_params.dirpath_app_wordlist,
            postfix_wordlist_app   = self.ut_params.postfix_app_wordlist,
            stopwords_list         = ['在', '年', '是', '说', '的', '和', '已经'],
            do_spelling_correction = False,
            do_word_stemming       = lf.LangFeatures().have_verb_conjugation(lang=lang),
            do_profiling           = False
        )
        text = [
            # Article 1
            '人工智能 ： 英 、 中 、 美 上演 “ 三国演义 ”',
            '英国 首相 特里莎·梅 周四 （1月 25日） 在 瑞士 达沃斯 世界 经济 论坛 上 宣布 ， 英国 在 人工智能 （ AI ） 领域 要 争 当 世界 领头羊。',
            '一周 后 ， 她 将 率 英国 经贸 代表团 访 华 ， 到 北京 和 上海 展开 " 历史性 访问 "。 一周 前 ， 中国 发表 《 人工智能 标准化 白皮书 》。',
            '中国 媒体 把 2017 年 称为 " AI 年 "， 2018 则 是 AI 从 学术 飞入 产业 、 普及 应用 的 关键 年 。',
            '围绕 AI ， 中美 正 胶着 于 争霸 竞赛 ，而 中英 在 科技 、工商 和 金融界 的 互动 将 产生 怎样 的 结果 ，引 人 关注'
            '。',
            # Article 2
            '叙利亚 俄军 遇袭 恐怖分子 用 无人机 “ 群攻 ”',
            '俄军 在 叙利亚 军事基地 遭到 攻击 后 ， 俄罗斯 国防部 警告 说 ， 恐怖分子 已 获得 先进 无人机 技术 ， 能够 在 全世界 发动 攻击 。',
            '俄罗斯 总参谋部 无人机 部门 负责人 亚历山大 · 维科夫 少将 说 ， 恐怖分子 使用 无人机 发动 攻击 的 威胁 已经 不再 是 不可能 的 事情，',
            '恐怖分子 已经 利用 无人机 攻击 俄军 在 叙利亚 的 克 美 明 空军基地 和 塔尔图斯 的 一个 港口',
            '他 还 说 ， 在 1月 6日 发动 攻击 的 技术 评估 显示 ，" 在 世界 所有 其他 地方 使用 无人机 发动 恐怖 攻击 已经 成为 现实 威胁"'
            # Article 3
        ]

        text_sentences_arr = [self.txt_preprocessor.process_text(inputtext=x) for x in text]
        log.Log.debugdebug(
            'PRE-PROCESSED ' + str(lang) + ' SENTENCES:\n\r' + str(text_sentences_arr)
        )

        # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
        # do_clustering(text=text, stopwords=stopwords, ncenters=2, freq_measure='tf', weigh_idf=False, verbose=0)
        res_cluster = self.do_clustering(
            text                   = text_sentences_arr,
            ncenters               = 2,
            expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9)),
            test_threshold_inside  = 0.7,
            test_threshold_outside = 0.3,
            test_description       = '1. ' + str(lang) + ' normalized, no IDF',
            freq_measure           = 'normalized',
            weigh_idf              = False
        )
        res.update(other_res_obj=res_cluster)
        # do_clustering(text=text, stopwords=stopwords, ncenters=2, freq_measure='frequency', weigh_idf=False, verbose=0)

        # Now weigh IDF
        log.Log.debugdebug('Weighing by IDF..')
        res_cluster = self.do_clustering(
            text                   = text_sentences_arr,
            ncenters               = 2,
            expected_clusters      = ((0,1,2,3,4),(5,6,7,8,9)),
            test_threshold_inside  = 0.7,
            test_threshold_outside = 0.3,
            test_description       = '2. ' + str(lang) + ' normalized, with IDF',
            freq_measure           = 'normalized',
            weigh_idf              = True
        )
        res.update(other_res_obj=res_cluster)

        return res

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        # res = self.test_textcluster_english()
        # res_final.update(other_res_obj=res)

        res = self.test_textcluster_chinese()
        res_final.update(other_res_obj=res)

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': PASSED ' + str(res_final.count_ok) + ', FAILED ' + str(res_final.count_fail)
        )

        return res_final


if __name__ == '__main__':
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = Config,
        default_config_file = Config.CONFIG_FILE_PATH_DEFAULT
    )
    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = None
    )

    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    TextClusterBasicUnitTest(
        ut_params = ut_params
    ).run_unit_test()
