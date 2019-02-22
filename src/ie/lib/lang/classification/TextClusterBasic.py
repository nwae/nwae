# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import re
import numpy as np
import pandas as pd
import ie.lib.math.Cluster as clst
import collections
import ie.lib.lang.characters.LangCharacters as lc
import ie.lib.util.StringUtils as su
import ie.lib.lang.model.FeatureVector as fv
import ie.lib.util.Log as log


#
# This is a basic text clustering algorithm, assuming that text is already nicely
# segmented into words (e.g. '我 昨天 的 流水 是 多少 ？'), uses frequency only as features,
# and relies on stopwords (and/or IDF measure) to increase accuracy.
# For English/Indonesian/Korean or any language with conjugations, root word extraction is a must,
# otherwise it will be inaccurate. Chinese, Thai, Vietnamese don't need this extra complication.
# TODO: The IDF word relevance weights are not efficient for the demos below, why?
# TODO: Normalized frequency works well for Chinese, but not English, why?
#
# NOTE: This Class is NOT Thread Safe
#
class TextClusterBasic:

    #
    # Initialize with a list of text, assumed to be already word separated by space.
    # TODO: Can't default to just [space] as word separator for languages like Vietnamese.
    #
    def __init__(self, text, stopwords):
        self.text_original = text
        self.text = text
        # Since we use automated IDF already, stopwords are not really needed anymore
        self.stopwords = stopwords
        # Basic pre-processing
        self.__preprocessing__()

        # First we need the keywords for feature vector, then the sentence matrix based on these keywords
        self.df_keywords_for_fv = None
        self.keywords_for_fv = None
        self.df_sentence_matrix = None
        self.sentence_matrix = None
        self.df_idf = None
        self.idf_matrix = None
        return

    def __preprocessing__(self):
        for i in range(0, len(self.text), 1):
            s = self.text[i]
            if type(s) != type('string'):
                s = str(s)
                raise Exception(str(self.__class__) + 'Warning line ' + str(i) + ', not string [' + s + ']')
            self.text[i] = su.StringUtils.trim(s.lower())
        return

    #
    # Option to pre-define keywords instead of extracting from text
    #
    def set_keywords(self, df_keywords):
        self.df_keywords_for_fv = df_keywords
        self.keywords_for_fv = list(df_keywords['Word'])
        return

    #
    # TODO: Add POS tagging to word segmentation so that POS tag will be another feature,
    # TODO: and use some kind of keyword extraction algorithm.
    # TODO: TextRank don't work very well at all, need something else.
    #
    def calculate_top_keywords(self, remove_quartile=50, verbose=0):
        log.Log.log('Calculating Top Keywords. Using the following stopwords:')
        log.Log.log(self.stopwords)

        #
        # 2. Get highest frequency words from the split sentences, and remove stop words
        #
        # Paste all sentences into a single huge vector
        all_words = ''
        for i in range(0, len(self.text), 1):
            all_words = all_words + ' ' + su.StringUtils.trim(self.text[i])
        all_words_split = all_words.split(' ')
        # Remove empty string ''
        all_words_split = [ x for x in all_words_split if x!='' ]

        # Remove numbers, or just fullstop/commas
        df_tmp = pd.DataFrame({'Word':all_words_split})
        is_number = df_tmp['Word'].str.match(pat='^[0-9.,]+$', case=False)
        df_tmp = df_tmp[~is_number]
        all_words_split = list(df_tmp['Word'])

        col = collections.Counter(all_words_split)
        # Order by top frequency keywords, and also convert to a Dictionary type (otherwise we can't extract
        # into DataFrame columns later)
        col = col.most_common()
        df_word_freq_all = pd.DataFrame({'Word': [x[0] for x in col], 'Frequency': [x[1] for x in col]})

        df_word_freq = pd.DataFrame()

        for i in range(0, df_word_freq_all.shape[0], 1):
            word = df_word_freq_all['Word'][i]
            freq = df_word_freq_all['Frequency'][i]

            # Remove word/sentence separators, punctuations
            if (word in lc.LangCharacters.UNICODE_BLOCK_PUNCTUATIONS)\
                    or (word in lc.LangCharacters.UNICODE_BLOCK_WORD_SEPARATORS)\
                    or (word in lc.LangCharacters.UNICODE_BLOCK_SENTENCE_SEPARATORS):
                continue
            # Clean up punctuations at the end of a word
            word = re.sub('[.,:;\[\]{}\-"'']$', '', word)
            #
            # Remove stopwords, quite an important step to improve clustering accuracy.
            # TODO: Once we have an efficient keyword extraction algorithm, we won't need stopwords anymore.
            #
            if (word in self.stopwords):
                if verbose >= 2:
                    print('Stopword [' + word + '] ignored..')
                continue

            df_word_freq = df_word_freq.append(pd.DataFrame({'Word':[word], 'Frequency':[freq]}), ignore_index=True)

        df_word_freq['Prop'] = df_word_freq['Frequency'] / sum(df_word_freq['Frequency'])
        #if verbose >= 1:
        #    print(df_word_freq)

        #
        # There will be a lot of words, so we remove (by default) the lower 50% quartile of keywords.
        # This will help wipe out a lot of words, up to 80-90% or more.
        #
        q_remove = 0
        # If user passes in 0, no words will be removed
        if remove_quartile > 0:
            q_remove = np.percentile(df_word_freq['Frequency'], remove_quartile)
        df_word_freq_qt = df_word_freq[df_word_freq['Frequency'] > q_remove]
        df_word_freq_qt = df_word_freq_qt.reset_index(drop=True)
        #if verbose >= 1:
        #    print(df_word_freq_qt)

        self.df_keywords_for_fv = df_word_freq_qt
        self.keywords_for_fv = list(df_word_freq_qt['Word'])

        return


    def calculate_sentence_matrix(self,
                                  freq_measure='tf',
                                  feature_presence_only=False,
                                  idf_matrix = None,
                                  verbose=0):
        #
        # 3. Model the sentences into a feature vector, using word frequency, relative positions, etc. as features
        #
        # Using the keywords set in this class, we create a profile template
        #if verbose >= 1:
        #    print(self.keywords_for_fv)
        no_keywords = len(self.keywords_for_fv)

        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=self.keywords_for_fv)
        #if verbose >= 1:
        #    print(model_fv.fv_template)

        #
        # 4. Link top keywords to sentences in a matrix, used as features in feature vector
        #
        # Create a frequency matrix of keywords by sentences
        # Get feature vector of sentence
        # Number of rows or rank (number of axes) in numpy speak
        nrow = len(self.text)
        ncol = no_keywords
        # By default this is TF
        sentence_matrix = np.zeros((nrow, ncol))
        sentence_matrix_freq = np.zeros((nrow, ncol))
        # The normalized version, by proportion
        sentence_matrix_norm = np.zeros((nrow, ncol))

        # Fill matrix
        for i in range(0, sentence_matrix.shape[0], 1):
            sent = su.StringUtils.trim( self.text[i] )
            if len(sent) == 0:
                continue
            df_fv = model_fv.get_freq_feature_vector(str=sent, feature_as_presence_only=feature_presence_only)
            # By default, use TF
            sentence_matrix[i] = list(df_fv['TF'])
            sentence_matrix_freq[i] = list(df_fv['Frequency'])
            sentence_matrix_norm[i] = list(df_fv['FrequencyNormalized'])
            if verbose >= 2:
                print('Sentence ' + str(i) + ': [' + sent + ']')
                #print(df_fv)
                #print(sentence_matrix[i])
                #print(sentence_matrix_norm[i])

        #
        # By default, we should use the TF form.
        #
        if freq_measure == 'normalized':
            sentence_matrix = sentence_matrix_norm
        elif freq_measure == 'frequency':
            sentence_matrix = sentence_matrix_freq

        if verbose >= 2:
            log.Log.log(str(self.__class__) + ' Sentence matrix (type=' + str(type(sentence_matrix)) + '):')
            log.Log.log(sentence_matrix)

        #
        # Weigh by IDF Matrix if given
        #
        if idf_matrix is not None:
            dim_sentence_matrix = sentence_matrix.shape
            dim_idf_matrix = idf_matrix.shape
            if dim_idf_matrix[0]!=dim_sentence_matrix[1]:
                raise Exception('IDF Matrix Dimensions must be (N,1) where N=number of columns in sentence matrix. ' +
                                'Sentence Matrix Dimension = ' + str(dim_sentence_matrix) + '. ' +
                                'IDF Matrix Dimension = ' + str(dim_idf_matrix))
            # This is not matrix multiplication, but should be
            sentence_matrix = np.multiply(sentence_matrix, np.transpose(idf_matrix))
            # Normalize back if initially using a normalized measure
            if freq_measure == 'normalized':
                for j in range(0, sentence_matrix.shape[0], 1):
                    if verbose >= 2:
                        log.Log.log('For sentence ' + str(j))
                        log.Log.log(sentence_matrix[j])

                    normalize_factor = np.sum(np.multiply(sentence_matrix[j], sentence_matrix[j])) ** 0.5

                    #log.Log.log('Normalize factor = ' + str(normalize_factor))
                    if normalize_factor > 0:
                        sentence_matrix[j] = sentence_matrix[j] / normalize_factor

            if verbose >= 2:
                log.Log.log(str(self.__class__) + ' After IDF weights, sentence matrix:')
                log.Log.log(sentence_matrix)

        # Convert to dataframe with names
        df_sentence_matrix = pd.DataFrame(data=sentence_matrix, columns=self.keywords_for_fv)
        self.df_sentence_matrix = df_sentence_matrix
        self.sentence_matrix = np.matrix(df_sentence_matrix.values)

        return

    #
    # Inverse Document Frequency = log(Total_Documents / Number_of_Documents_of_Word_occurrence)
    # This measure doesn't work well in automatic clustering of texts.
    #
    def calculate_idf(self, verbose=0):
        word_presence_matrix = (self.sentence_matrix>0)*1
        # Sum columns
        keyword_presence = np.sum(word_presence_matrix, axis=0)
        # Make sure idf is not infinity
        keyword_presence[keyword_presence==0] = 1
        # Total document count
        n_documents = self.sentence_matrix.shape[0]
        idf = np.log(n_documents / keyword_presence)
        df_idf = None
        try:
            if verbose >= 2:
                log.Log.log(str(self.__class__) + ' Using Keywords: ' + str(self.keywords_for_fv))
                log.Log.log(str(self.__class__) + ' Using IDF type ' + str(type(idf)) + ': ' + str(idf))
            # To make sure idf is in a single array, we convert to values as it could be a single row narray or matrix
            df_idf = pd.DataFrame({'Word':self.keywords_for_fv, 'IDF':idf.tolist()[0]})
        except Exception as ex:
            log.Log.log(str(self.__class__) + ' Error for IDF data frame.')
            log.Log.log(ex)
            raise(ex)
        #if verbose >= 1:
        #    print(df_idf)
        self.df_idf = df_idf
        # Make sure to transpose to get a column matrix
        self.idf_matrix = np.transpose( np.matrix(df_idf['IDF'].values) )

        if verbose >= 2:
            log.Log.log(str(self.__class__) + 'IDF Matrix as follows:')
            log.Log.log(self.idf_matrix)
        return

    #
    # The main clustering function
    #
    def cluster(self,
                ncenters,
                iterations=50,
                feature_presence_only=False,
                freq_measure='tf',
                weigh_idf=False,
                optimal_cluster_threshold_change=clst.Cluster.THRESHOLD_CHANGE_DIST_TO_CENTROIDS,
                verbose=0):
        # Model sentences into feature vectors
        self.calculate_sentence_matrix(freq_measure=freq_measure,
                                       feature_presence_only=feature_presence_only,
                                       idf_matrix=None,
                                       verbose=verbose)

        #
        # From the sentence matrix, we can calculate the IDF
        #
        # Do a redundant multiplication so that a copy is created, instead of pass by reference
        sentence_matrix = self.sentence_matrix * 1
        if weigh_idf:
            self.calculate_idf(verbose=verbose)
            # Recalculate sentence matrix
            self.calculate_sentence_matrix(freq_measure=freq_measure,
                                           feature_presence_only=feature_presence_only,
                                           idf_matrix=self.idf_matrix.copy(),
                                           verbose=verbose)

        #
        # 5. Cluster sentences in matrix
        #
        # Optimal clusters
        # TODO: Improve the optimal cluster algorithm before using the value returned
        if False:
            optimal_clusters = clst.Cluster.get_optimal_cluster(
                matx=sentence_matrix,
                n_tries=10,
                iterations=iterations,
                threshold_change=optimal_cluster_threshold_change,
                verbose=verbose)
            if verbose>=1:
                log.Log.log('Optimal Clusters = ' + str(optimal_clusters))

        retval_cluster = clst.Cluster.cluster(matx=sentence_matrix,
                                              feature_names=self.keywords_for_fv,
                                              ncenters=ncenters,
                                              iterations=iterations,
                                              verbose=1)
        df_cluster_matrix = retval_cluster['ClusterMatrix']
        df_point_cluster_info = retval_cluster['PointClusterInfo']

        # Add column of cluster number to text data frame
        df_point_cluster_info['Text'] = self.text
        return { 'ClusterMatrix': df_cluster_matrix, 'TextClusterInfo': df_point_cluster_info }


    #
    # Unit Test for TextClusterBasic Class
    #
    class UnitTest:
        ######################################
        # Test Functions below
        #   One observation is that the TF measure is quite bad, and using our normalized form is much better.

        def __init__(self, verbose=0):
            self.verbose = verbose
            return

        def do_clustering(self,
                          text,
                          stopwords,
                          ncenters,
                          feature_presence_only=False,
                          freq_measure='tf',
                          weigh_idf=False):

            tc = TextClusterBasic(text, stopwords)
            tc.calculate_top_keywords(remove_quartile=50, verbose=self.verbose)

            retval = tc.cluster(ncenters      = ncenters,
                                iterations    = 50,
                                freq_measure  = freq_measure,
                                weigh_idf     = weigh_idf,
                                feature_presence_only = feature_presence_only,
                                verbose=self.verbose)
            df_text_cluster = retval['TextClusterInfo']
            print(list(df_text_cluster['ClusterNo']))

            return

        def test_textcluster_english(self):
            print('English Demo')
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
            stopwords = [
                'the', 'of', 'in', 'on', 'and', 'or', 'to', 'be', 'a', 'is', 'are', 'at', 'as', 'for', 'this', 'that',
                'was', 'were', 'which', 'when', 'where', 'will', 'would', 'with', 'his', 'her', 'it', 'from', 'than',
                'who', 'while', 'they', 'could', 'these', 'those', 'has', 'have', 'through', 'some', 'other', 'way'
            ]

            # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
            # do_clustering(text=text, stopwords=stopwords, ncenters=3, freq_measure='tf', weigh_idf=False, verbose=0)
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=3,
                freq_measure='normalized',
                weigh_idf=False
            )
            # do_clustering(text=text, stopwords=stopwords, ncenters=3, freq_measure='frequency', weigh_idf=False, verbose=0)

            # Now weigh IDF
            print('Weighing by IDF..')
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=3,
                freq_measure='normalized',
                weigh_idf=True
            )

            print('Now using only feature presence (means freq is 0 or 1 only)')
            #
            # Word presence only
            #
            # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
            # do_clustering(text=text, stopwords=stopwords, ncenters=3, feature_presence_only=True, freq_measure='tf', weigh_idf=False, verbose=0)
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=3,
                feature_presence_only=True,
                freq_measure='normalized',
                weigh_idf=False
            )
            # do_clustering(text=text, stopwords=stopwords, ncenters=3, feature_presence_only=True, freq_measure='frequency', weigh_idf=False, verbose=0)

            # Now weigh IDF
            print('Weighing by IDF..')
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=3,
                feature_presence_only=True,
                freq_measure='normalized',
                weigh_idf=True
            )
            return

        def test_textcluster_chinese(self):
            print('Chinese Demo')
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
            stopwords = [
                '在', '年', '是', '说', '的', '和', '已经'
            ]

            # This example is too small in sample size to weigh by IDF (which will instead lower the accuracy)
            # do_clustering(text=text, stopwords=stopwords, ncenters=2, freq_measure='tf', weigh_idf=False, verbose=0)
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=2,
                freq_measure='normalized',
                weigh_idf=False
            )
            # do_clustering(text=text, stopwords=stopwords, ncenters=2, freq_measure='frequency', weigh_idf=False, verbose=0)

            # Now weigh IDF
            print('Weighing by IDF..')
            self.do_clustering(
                text=text,
                stopwords=stopwords,
                ncenters=2,
                freq_measure='normalized',
                weigh_idf=True
            )
            return

        def run_tests(self):
            self.test_textcluster_english()
            self.test_textcluster_chinese()
            return


if __name__ == '__main__':
    ut = TextClusterBasic.UnitTest(verbose=1)
    ut.run_tests()
