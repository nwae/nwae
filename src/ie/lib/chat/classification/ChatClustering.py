#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import os
import pandas as pd
import collections
import ie.app.ConfigFile as cf
import ie.lib.util.StringUtils as su
import ie.lib.chat.Chat as chat
import ie.lib.lang.nlp.WordList as wl
import ie.lib.lang.nlp.WordSegmentation as ws
import ie.lib.lang.nlp.SynonymList as sl
import ie.lib.lang.classification.TextClusterBasic as tc
import ie.lib.chat.bot.Intent as lb


#
# Purpose:
#   1. Automatically clusters chats (using member line 1) to determine question types,
#      automatic summaries, etc.
#   2. Also as a way to collect training material for out chatbot (LeBot).
# Steps of clustering:
#   1. Extract Member Line 1,2,3.. from chat data.
#      TODO: Extract intelligence also from the other spoken lines.
#   2. Cluster member spoken line 1, if it is a greeting use 2nd line, otherwise 1st line.
#   3. Do word segmentation on content. Common dictionary wordlists and user wordlists
#      need to be maintained.
#      User wordlist here:
#      https://drive.google.com/open?id=1w8zOp_Aue0x7ZwZFMO7AX6I5Mc0EBd4jLB3MHxZ8ZrA
#      TODO: All this wordlists need to be in DB, and a BO webpage to maintain it.
#   4. Sentence/Text is represented as a quantitative math object (matrix, vector, etc.)
#      that is computable (i.e. clusterable)
#      TODO: Include other key features, not just words
#
# A Note on Stopwords:
#   For our training and intent/command classification, we do not use stopwords but a
#   keyword measure based on IDF, which simplifies maintenance.
#   However for clustering, unless we do POS Tagging, this will be inaccurate.
#
class ChatClustering:

    # Because we remove greeting lines if the 1st spoken line is a greeting by member, we need the name of the intent
    INTENT_GREETING = 'common.greeting'

    def __init__(
            self,
            lang,
            brand,
            currency,
            datefrom,
            dateto,
            lebot = None,
            chatdata_date_format='%Y-%m-%d %H:%M:%S',
    ):

        self.brand = brand
        self.currency = currency
        self.datefrom = datefrom
        self.dateto = dateto
        self.lebot = lebot
        self.chatdata_date_format = chatdata_date_format

        self.lang = lang
        self.wseg = ws.WordSegmentation(lang             = self.lang,
                                        dirpath_wordlist = cf.ConfigFile.DIR_WORDLIST,
                                        postfix_wordlist = cf.ConfigFile.POSTFIX_WORDLIST)

        # Synonyms
        self.synonymlist = sl.SynonymList(lang                = self.lang,
                                          dirpath_synonymlist = cf.ConfigFile.DIR_SYNONYMLIST,
                                          postfix_synonymlist = cf.ConfigFile.POSTFIX_SYNONYMLIST)
        self.synonymlist.load_synonymlist(verbose=1)

        self.calllog = None

        # Load application wordlist

        # Add application wordlist
        self.wseg.add_wordlist(dirpath = cf.ConfigFile.DIR_APP_WORDLIST,
                               postfix='.'+self.brand+cf.ConfigFile.POSTFIX_APP_WORDLIST,
                               verbose=1)
        # Add application stopwords
        self.stopwords = wl.WordList(lang = self.lang,
                                     dirpath_wordlist = cf.ConfigFile.DIR_APP_STOPWORDS,
                                     postfix_wordlist=cf.ConfigFile.POSTFIX_APP_STOPWORDS)
        self.stopwords.load_wordlist(verbose=1)

        # We need this for text clustering
        self.stopwords_list = list( self.stopwords.wordlist['Word'] )
        self.textcluster = None

        # Add synonym list to wordlist (just in case they are not synched)
        len_before = self.wseg.lang_wordlist.wordlist.shape[0]
        self.wseg.add_wordlist(dirpath=None, postfix=None, array_words=list(self.synonymlist.synonymlist['Word']))
        len_after = self.wseg.lang_wordlist.wordlist.shape[0]
        if len_after - len_before > 0:
            print("Warning. These words not in word list but in synonym list:")
            words_not_synched = self.wseg.lang_wordlist.wordlist['Word'][len_before:len_after]
            print(words_not_synched)

        # Filepath to member first non-greeting line
        self.fpath_memberlogs_linetop = cf.ConfigFile.DIR_CHATCLUSTERING_OUTPUT + '/' +\
                                        self.brand + "." + self.currency + "." +\
                                        self.datefrom + ".to." + self.dateto + ".membervisitor.line.01.csv"

        return

    def load_data(
            self,
            maxlines=0,
            verbose=0
    ):
        self.calllog = chat.Chat(
            lang        = self.lang,
            brand       = self.brand,
            currency    = self.currency,
            datefrom    = self.datefrom,
            dateto      = self.dateto,
            date_format = self.chatdata_date_format
        )
        self.calllog.get_data_from_file(dirpath=cf.ConfigFile.DIR_CHATDATA, maxlines=maxlines, verbose=verbose)
        if verbose >= 1:
            print('Loaded ' + str(self.calllog.chatdata.shape[0]) + ' lines of chat data')
        return

    #
    # Instead of the whole chat, we extract the first member non-greeting line from every chat
    #
    def preprocess_chatdata_into_member_first_non_greeting_line(
            self,
            maxlines=0,
            verbose=0
    ):

        self.load_data(
            maxlines             = maxlines,
            verbose              = verbose
        )
        # Get only member spoken lines
        memberlogs_sort = self.calllog.get_member_spoken_lines(label_first_n=2, verbose=1)

        #
        # Now we analyze all line 1, the majority of which is the chat question itself (about 50%).
        #
        tmp = memberlogs_sort.loc[memberlogs_sort[chat.Chat.COL_MEMBER_CHAT_LINE_NO] > 0]
        memberlogs_linetop = pd.DataFrame({
            chat.Chat.COL_ID: tmp[chat.Chat.COL_ID],
            chat.Chat.COL_SPEAKER_TYPE: tmp[chat.Chat.COL_SPEAKER_TYPE],
            chat.Chat.COL_SPEAKER_NAME: tmp[chat.Chat.COL_SPEAKER_NAME],
            chat.Chat.COL_CONTENT: tmp[chat.Chat.COL_CONTENT],
            chat.Chat.COL_CHAT_LINE_NO: tmp[chat.Chat.COL_CHAT_LINE_NO],
            chat.Chat.COL_MEMBER_CHAT_LINE_NO: tmp[chat.Chat.COL_MEMBER_CHAT_LINE_NO]})
        memberlogs_linetop = memberlogs_linetop.reset_index(drop=True)

        #
        # 1. First step is always word segmentation
        #
        nlines = memberlogs_linetop.shape[0]
        memberlogs_linetop[chat.Chat.COL_CONTENT_SPLIT] = memberlogs_linetop[chat.Chat.COL_CONTENT]
        for line in range(0, nlines, 1):
            chatline = memberlogs_linetop[chat.Chat.COL_CONTENT].loc[line]
            chatline = su.StringUtils.trim(chatline)
            chatline_segmented = self.wseg.segment_words(text=chatline)
            if verbose >= 1:
                print('Line ' + str(line) + ' (of ' + str(nlines) + ' lines): + "' + chatline + '"')
                print('          Segmented Text: ' + '"' + chatline_segmented + '"')
            memberlogs_linetop[chat.Chat.COL_CONTENT_SPLIT].at[line] = chatline_segmented

        #
        # We use LeBot to classify all lines first, so we can remove greeting lines
        #
        memberlogs_linetop[chat.Chat.COL_INTENT] = ['-'] * memberlogs_linetop.shape[0]
        memberlogs_linetop[chat.Chat.COL_INTENT_SCORE] = [0.0] * memberlogs_linetop.shape[0]
        memberlogs_linetop[chat.Chat.COL_INTENT_SCORE_CONFIDENCE] = [0.0] * memberlogs_linetop.shape[0]

        # print(memberlogs_linetop)

        for line in range(0, memberlogs_linetop.shape[0], 1):
            str_split = memberlogs_linetop[chat.Chat.COL_CONTENT_SPLIT].loc[line]
            member = memberlogs_linetop[chat.Chat.COL_SPEAKER_NAME].loc[line]

            df_result = self.lebot.get_text_class(
                text_segmented            = str_split,
                weigh_idf                 = True,
                top                       = lb.Intent.SEARCH_TOPX_RFV,
                return_match_results_only = True,
                score_min_threshold       = lb.Intent.CONFIDENCE_LEVEL_1_SCORE,
                verbose                   = 0
            )

            msg = 'Line ' + str(line) + ': (speaker=' + member + ')'
            if df_result is None:
                msg = msg + ' [' + str_split + '] = [NOMATCH]'
            else:
                # Replace with normalized text (LeBot will replace synonyms with rootwords)
                str_split_normalized = df_result[lb.Intent.COL_TEXT_NORMALIZED].loc[0]
                memberlogs_linetop[chat.Chat.COL_CONTENT_SPLIT].at[line] = str_split_normalized
                # Extract results from LeBot
                top_result = df_result[lb.Intent.COL_COMMAND].loc[0]
                top_result_match = df_result[lb.Intent.COL_MATCH].loc[0]
                top_result_score = df_result[lb.Intent.COL_SCORE].loc[0]
                top_result_score_conf = df_result[lb.Intent.COL_SCORE_CONFIDENCE_LEVEL].loc[0]

                msg = msg + ' [' + str_split_normalized + '] = [' + top_result +\
                      ', score=' + str(top_result_score) + '， conflevel=' + str(top_result_score_conf) + ']'

                if top_result_match == 1:
                    memberlogs_linetop[chat.Chat.COL_INTENT].at[line] = top_result
                    memberlogs_linetop[chat.Chat.COL_INTENT_SCORE].at[line] = top_result_score
                    memberlogs_linetop[chat.Chat.COL_INTENT_SCORE_CONFIDENCE].at[line] = top_result_score_conf

            print(msg)

        #
        # Remove purely 'greeting' lines
        #
        print('Removing greeting lines, total lines before = ' + str(memberlogs_linetop.shape[0]))
        memberlogs_linetop = memberlogs_linetop[
            memberlogs_linetop[chat.Chat.COL_INTENT] != ChatClustering.INTENT_GREETING]
        print('Remain ' + str(memberlogs_linetop.shape[0]) + ' lines')

        col_chatid = collections.Counter(memberlogs_linetop[chat.Chat.COL_ID].values)
        # We only call most_common() to convert the data type into a dictionary so that we can extract
        # columns into a data frame, and for no other reason.
        col_chatid = col_chatid.most_common()
        # Order by top frequency keywords
        tmp_df = pd.DataFrame({chat.Chat.COL_ID: [x[0] for x in col_chatid],
                               'ChatIDLineCount': [x[1] for x in col_chatid]})
        # Merge with Chat ID line count
        memberlogs_linetop = memberlogs_linetop.merge(right=tmp_df, on=[chat.Chat.COL_ID])
        memberlogs_linetop = memberlogs_linetop.sort_values(by=[chat.Chat.COL_ID, chat.Chat.COL_MEMBER_CHAT_LINE_NO],
                                                            ascending=True)
        # print(memberlogs_linetop[0:10])

        # Remove line 2 of the same Chat ID if there is already line 1
        is_chatid_member_line_2 = (memberlogs_linetop[chat.Chat.COL_MEMBER_CHAT_LINE_NO] == 2)
        have_2_lines = (memberlogs_linetop['ChatIDLineCount'] == 2)
        memberlogs_linetop = memberlogs_linetop[~(is_chatid_member_line_2 & have_2_lines)]
        print('Total lines now = ' + str(memberlogs_linetop.shape[0]))

        # print(memberlogs_linetop[0:10])

        # Write to file
        if verbose >= 1: print('Writing member chat logs (first line) to file [' + self.fpath_memberlogs_linetop + ']')
        # This automatically escapes quotes and ensures that commas are unique as separators
        memberlogs_linetop.to_csv(self.fpath_memberlogs_linetop, sep=',')
        return

    def analyze(self, no_keywords=50, verbose=0):

        # Top lines by member (currently we use 1st and 2nd line)
        memberlogs_linetop = None
        df_stats = pd.DataFrame()

        if not os.path.isfile(self.fpath_memberlogs_linetop):
            if verbose >= 1:
                print("Can't find file [", self.fpath_memberlogs_linetop, "].")
                return
        else:
            print('Found file [' + self.fpath_memberlogs_linetop + ']. Getting data from file..')
            memberlogs_linetop = pd.read_csv(self.fpath_memberlogs_linetop, sep=',', index_col=0)

        # Get some curious stats
        df_stats['TotalChats'] = [ memberlogs_linetop.shape[0] ]
        df_stats['UniqueMembers'] = [ len( set(memberlogs_linetop[chat.Chat.COL_SPEAKER_NAME]) ) ]

        #
        # Create TextCluster object
        #
        self.textcluster = tc.TextClusterBasic(text=list(memberlogs_linetop[chat.Chat.COL_CONTENT_SPLIT]),
                                               stopwords=self.stopwords_list)

        #
        # 2. Get highest frequency words from the split sentences, and remove stop words
        #
        self.textcluster.calculate_top_keywords(remove_quartile=75, verbose=0)
        df_word_freq_75 = self.textcluster.df_keywords_for_fv
        if verbose >= 1:
            print(df_word_freq_75[0:30])

        # Keep top keywords to file
        fname_tmp = cf.ConfigFile.DIR_CHATCLUSTERING_OUTPUT + '/top-keywords.' + self.brand + '.' + self.currency + '.' +\
                    self.datefrom + '.to.' + self.dateto + '.csv'
        if verbose >= 1:
            print('Writing top keywords (75% quartile) to file [' + fname_tmp + ']..')
        df_word_freq_75.to_csv(fname_tmp, sep=',')

        # Total coverage of the top keywords of all words in the 1st line member spoken chats
        df_stats['TopKeywords'] = [ df_word_freq_75.shape[0] ]

        #
        # 3. Model the sentences into a feature vector, using word frequency, relative positions, etc. as features
        #
        # Using the above top keywords, we create a profile template
        words_for_fv = list(df_word_freq_75['Word'][0:no_keywords])
        df_stats['TotalWordCoverageTopKeywords'] = [ sum(df_word_freq_75['Prop']) ]

        retval_txtcluster = self.textcluster.cluster(ncenters = 15,
                                                   iterations = 20,
                                                   feature_presence_only = False,
                                                   freq_measure = 'normalized',
                                                   weigh_idf=False,
                                                   optimal_cluster_threshold_change=0.001,
                                                   verbose = 1)

        df_text_cluster = retval_txtcluster['TextClusterInfo']
        # Add column of cluster number to data frame
        memberlogs_linetop = memberlogs_linetop.assign(ClusterNo=list(df_text_cluster['ClusterNo']))
        memberlogs_linetop = memberlogs_linetop.assign(DistanceToCenter=list(df_text_cluster['DistanceToCenter']))

        counter = collections.Counter(list(memberlogs_linetop['ClusterNo']))
        counter = counter.most_common()
        df_cluster_freq = pd.DataFrame({'Cluster':[x[0] for x in counter], 'Freq':[x[1] for x in counter]})
        if verbose >= 1:
            print(df_cluster_freq)

        # Now to find within the clusters, it's top closest points to the cluster center
        topchats_in_clusters = pd.DataFrame()
        for cluster_no in list(df_cluster_freq['Cluster']):
            cluster_chatl1 = memberlogs_linetop[memberlogs_linetop['ClusterNo'] == cluster_no]
            # Order by distance to center increasing
            cluster_chatl1 = cluster_chatl1.sort_values(by=['DistanceToCenter'],ascending=[True])
            topchats_in_clusters = topchats_in_clusters.append(pd.DataFrame(cluster_chatl1), ignore_index=True)

        # Record to file top chats from clusters
        df_top_chats = pd.DataFrame({'ClusterNo': topchats_in_clusters['ClusterNo'],
                                     'DistanceToCenter': topchats_in_clusters['DistanceToCenter'],
                                     chat.Chat.COL_ID: topchats_in_clusters[chat.Chat.COL_ID],
                                     chat.Chat.COL_SPEAKER_TYPE: topchats_in_clusters[chat.Chat.COL_SPEAKER_TYPE],
                                     chat.Chat.COL_SPEAKER_NAME: topchats_in_clusters[chat.Chat.COL_SPEAKER_NAME],
                                     chat.Chat.COL_CHAT_LINE_NO: topchats_in_clusters[chat.Chat.COL_CHAT_LINE_NO],
                                     chat.Chat.COL_INTENT: topchats_in_clusters[chat.Chat.COL_INTENT],
                                     chat.Chat.COL_INTENT_SCORE: topchats_in_clusters[chat.Chat.COL_INTENT_SCORE],
                                     chat.Chat.COL_CONTENT_SPLIT: topchats_in_clusters[chat.Chat.COL_CONTENT_SPLIT]
                                     })
        fpath = cf.ConfigFile.DIR_CHATCLUSTERING_OUTPUT + '/' + 'top.chats.' +\
                self.brand + '.' + self.currency + '.' + self.datefrom + '.to.' + self.dateto + '.csv'
        df_top_chats.to_csv(fpath, sep=',')
        print('Wrote top chats to file [' + fpath + ']')

        if verbose >= 1:
            print(df_stats)

        return df_top_chats

