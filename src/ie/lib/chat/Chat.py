#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import pandas as pd
# Library to convert Traditional Chinese to Simplified Chinese
import hanziconv as hzc


#
# Represents our unified chat format as object
#
class Chat:

    COL_ID                  = 'CalllogID'
    COL_DATETIME_STR        = 'DateTime'
    # Derived column, after conversion to datetime object
    COL_DATETIME            = 'datetime'
    COL_SPEAKER_TYPE        = 'Category'
    COL_SPEAKER_NAME        = 'Speaker'
    COL_CHAT_LINE_NO        = 'Line'
    # Derived column, member chat line
    COL_MEMBER_CHAT_LINE_NO = 'MemberLine'
    COL_CONTENT             = 'Content'
    # Derived column, conversion to simplified chinese
    COL_CONTENT_SIMPLIFIED  = 'content.simplified'
    # Derived column, word splitting
    COL_CONTENT_SPLIT       = 'content.split'
    # Derived column, Command/Intent by LeBot
    COL_INTENT              = 'Intent'
    COL_INTENT_SCORE        = 'IntentScore'
    COL_INTENT_SCORE_CONFIDENCE = 'IntentScoreConfidence'

    def __init__(
            self,
            lang,
            brand,
            currency,
            datefrom,
            dateto,
            date_format = '%Y-%m-%d %H:%M:%S'
    ):
        self.lang = str(lang).lower()
        self.brand = str(brand).lower()
        self.currency = str(currency).lower()
        self.datefrom = datefrom
        self.dateto = dateto
        self.chatdata = pd.DataFrame()

        self.date_format = date_format
        return

    # If requested to convert to Simplified Chinese (will be slow!!)
    # However, from casual observation on our chats, almost 100% of the chats are all in Simplified Chinese,
    # meaning that this feature is probably rarely needed.
    # In fact from the millions of chat lines, I have never seen any with Traditional Chinese so far.
    def convert_to_simplified_chinese(self, verbose=0):
        self.chatdata[Chat.COL_CONTENT_SIMPLIFIED] = self.chatdata[Chat.COL_CONTENT]
        if verbose >= 1:
            print('Converting to simplified chinese..')
        for i in range(0, self.chatdata.shape[0], 1):
            text = str(self.chatdata.loc[i]['content'])
            self.chatdata.loc[i][Chat.COL_CONTENT_SIMPLIFIED] = hzc.HanziConv.toSimplified(text)
            if verbose >= 1:
                if i % 10000 == 0:
                    print('  Converted ' + str(i) + ' lines..')
        return

    #
    # Data comes from file which in turn comes from the R code in this folder ChatData.R.
    #
    def get_data_from_file(
            self,
            dirpath,
            convert_to_simplified_chinese=False,
            maxlines=0,
            verbose=0
    ):
        filepath =  self.brand + '.' + self.currency + '.' + self.datefrom + '.to.' + self.dateto + '.csv'

        try:
            self.chatdata = pd.read_csv(filepath_or_buffer=dirpath + '/' + filepath, sep=',', header=0)
            self.chatdata[Chat.COL_DATETIME] = pd.to_datetime(self.chatdata[Chat.COL_DATETIME_STR], format=self.date_format)
            if verbose>0: print('Read chat data [' + filepath + '], ' + self.chatdata.shape[0].__str__() + ' lines.')
        except IOError as e:
            print('Can\'t open file [' + dirpath + '/' + filepath + ']. ')
            raise e

        self.chatdata[Chat.COL_ID] = self.chatdata[Chat.COL_ID].astype(str)
        self.chatdata[Chat.COL_SPEAKER_NAME] = self.chatdata[Chat.COL_SPEAKER_NAME].astype(str)

        # Sort by speaker, datetime
        self.chatdata = self.chatdata.sort_values(by=[Chat.COL_ID,Chat.COL_DATETIME])
        self.chatdata = self.chatdata.reset_index(drop=True)

        if maxlines > 0 and self.chatdata.shape[0] >= maxlines:
            self.chatdata = self.chatdata[0:maxlines]

        if self.lang == 'cn' and convert_to_simplified_chinese:
            self.convert_to_simplified_chinese(verbose=verbose)

        return

    #
    # Retrieve only member spoken lines, with line numbers (up to 'label_first_n')
    # Currently used for our chat clustering purpose, because I can't cluster the whole chat.
    # So I only use the very first topics raised by member.
    #
    def get_member_spoken_lines(self, label_first_n=10, verbose=0):
        # Only filter member/visitor lines
        memberlogs = self.chatdata.loc[(self.chatdata[Chat.COL_SPEAKER_TYPE] == 'Member') |
                                       (self.chatdata[Chat.COL_SPEAKER_TYPE] == 'Visitor')]

        # Order by Live Session ID, Speaker, line
        memberlogs_sort = memberlogs.sort_values(by=[Chat.COL_ID, Chat.COL_CHAT_LINE_NO])
        # Unlike R, we need to manually reset row indexes
        memberlogs_sort = memberlogs_sort.reset_index(drop=True)
        # if verbose > 0: print(memberlogs_sort[0:10])

        #
        # Get 1st Line of Member chat
        #
        # Shift down 1 step
        len_tmp = memberlogs_sort.shape[0] - 1 - 1
        # print('Length=' + str(len_tmp))
        prev_id_col = pd.Series(data=[''] + list(memberlogs_sort[Chat.COL_ID].loc[0:len_tmp]))
        # Panda Series data type
        member_line_1 = (memberlogs_sort[Chat.COL_ID] != prev_id_col)
        memberlogs_sort[Chat.COL_MEMBER_CHAT_LINE_NO] = member_line_1 * 1

        #
        # Label lines 2, 3, ...
        #
        prev_line_col = member_line_1
        for line in range(2, label_first_n+1, 1):
            prev_line_shift_down_col = pd.Series(data=[False] + list(prev_line_col.loc[0:len_tmp]))
            # Condition to be (n+1)-th line:
            # First line must not be true, and shifted down line before is true (previous line is n-th line)
            # Since member_line_1 is fixed, we know where our first lines are. Since we are shifting down
            # the pre_line_shift_down_col in every loop, the moment it hits a member first line, it neutralizes to False.
            curline_col = (member_line_1==False) & (prev_line_shift_down_col==True)
            memberlogs_sort[Chat.COL_MEMBER_CHAT_LINE_NO] = memberlogs_sort[Chat.COL_MEMBER_CHAT_LINE_NO] + curline_col*line
            prev_line_col = curline_col

        return memberlogs_sort


def demo_1():
    dirpath = '/Users/mark.tan/svn/yuna/app.data/chatdata'
    lang = 'cn'
    brand = 'TBet'
    currency = 'CNY'
    datefrom = '2018-07-01'
    dateto = '2018-09-30'
    cd = Chat(
        lang=lang,
        brand=brand,
        currency=currency,
        datefrom=datefrom,
        dateto=dateto,
        date_format='%d/%m/%Y %H:%M'
    )

    tosimplified=False
    cd.get_data_from_file(dirpath=dirpath, convert_to_simplified_chinese=tosimplified, maxlines=10000, verbose=1)

    if tosimplified:
        traditional_cn = cd.chatdata[cd.chatdata[Chat.COL_CONTENT] != cd.chatdata[Chat.COL_CONTENT_SIMPLIFIED]]
        print('Lines with traditional chinese = ' + str(traditional_cn.shape[0]))
        print(traditional_cn)

    if not cd.chatdata is None:
        print('Loaded chat data of ' + str(cd.chatdata.shape[0]) + ' lines.')
        print(cd.chatdata[50:100])

    memberlines = cd.get_member_spoken_lines(label_first_n=20, verbose=1)
    print(memberlines.loc[0:50])
    return


if __name__ == '__main__':
    demo_1()