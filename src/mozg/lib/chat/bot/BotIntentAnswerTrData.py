# -*- coding: utf-8 -*-

import os.path
import re
import pandas as pd
import numpy as np
import collections
import ie.lib.util.StringUtils as su
import ie.lib.util.Log as log


#
# TODO Read Intent, Answer, Training Data from DB instead
#
class BotIntentAnswerTrData:

    COL_BRAND          = 'Brand'
    COL_LANG           = 'Lang'
    # Category has folder like syntax
    COL_CATEGORY       = 'Category'
    COL_WEIGHT         = 'Weight'
    # Can be "system" (ignore), "regex" (regular expression type), "user" (what we want)
    COL_INTENT_TYPE    = 'Intent Type'
    COL_INTENT         = 'Intent'
    COL_INTENT_ID      = 'Intent ID'    # Derived column, joining Category + Intent
    COL_ANSWER         = 'Answer'
    # The training data is lumped together separated by newlines, which we need to break up
    COL_TRAINING_DATA  = 'Training Data'
    # After we split out the training data
    COL_TEXT           = 'Text'

    # Intent Types
    INTENT_TYPE_SYSTEM   = 'system'
    INTENT_TYPE_REGEX    = 'regex'
    INTENT_TYPE_USER     = 'user'
    # Internal types are for internal usage, e.g. CX team intents to classify chats.
    INTENT_TYPE_INTERNAL = 'internal'
    TRAINING_INTENT_TYPES = [INTENT_TYPE_USER, INTENT_TYPE_INTERNAL]

    # System Default Intents: Must exist intents
    INTENT_WELCOME  = 'common/welcome'
    INTENT_NOANSWER = 'common/noanswer'

    def __init__(self,
                 lang,
                 brand,
                 dirpath,
                 postfix_intent_answer_trdata,
                 postfix_trdata):
        self.brand = brand
        self.lang = lang
        self.postfix_trdata = postfix_trdata
        self.filepath_intent_answer_trdata = dirpath + '/' + lang + '.' + brand + '.' + postfix_intent_answer_trdata + '.csv'
        # Option to add internal intents
        self.filepath_intent_answer_trdata_internal = dirpath + '/' + lang + '.' + 'internal' + '.' + postfix_intent_answer_trdata + '.csv'

        # We will extract training data to this file, line by line instead of one whole chunk
        self.filepath_trdata = dirpath + '/' + lang + '.' + brand + '.' + postfix_trdata + '.csv'
        # For Comm100
        self.filepath_intent_answer_trdata_comm100 = dirpath + '/comm100.' + lang + '.' + brand + '.' + postfix_intent_answer_trdata + '.csv'

        try:
            log.Log.log('Reading Intent-Answer-Training Data from file [' + self.filepath_intent_answer_trdata + ']...')
            self.bot_replies = pd.read_csv(filepath_or_buffer=self.filepath_intent_answer_trdata, sep=',')
            log.Log.log('Read ' + str(self.bot_replies.shape[0]) + ' lines from file.')

            if os.path.isfile(self.filepath_intent_answer_trdata_internal):
                log.Log.log(
                    'Reading Internal Intents file [' + self.filepath_intent_answer_trdata_internal + ']...')
                self.bot_replies_internal = pd.read_csv(filepath_or_buffer=self.filepath_intent_answer_trdata_internal, sep=',')
                log.Log.log('Read ' + str(self.bot_replies_internal.shape[0]) + ' lines from internal intents')

                log.Log.log('Concatenating data frames...')
                self.bot_replies = pd.concat([self.bot_replies, self.bot_replies_internal])
                self.bot_replies = self.bot_replies.reset_index(drop=True)
                log.Log.log('Length now ' + str(self.bot_replies.shape[0]))

            # Sort the raw training data by Category/Intent
            self.bot_replies = self.bot_replies.sort_values(
                by        = [BotIntentAnswerTrData.COL_CATEGORY, BotIntentAnswerTrData.COL_INTENT],
                ascending = True
            )
            # Convert some columns to lowercase
            self.bot_replies[BotIntentAnswerTrData.COL_BRAND] = self.bot_replies[BotIntentAnswerTrData.COL_BRAND].str.lower()
            self.bot_replies[BotIntentAnswerTrData.COL_LANG] = self.bot_replies[BotIntentAnswerTrData.COL_LANG].str.lower()
            self.bot_replies[BotIntentAnswerTrData.COL_INTENT_TYPE] = self.bot_replies[BotIntentAnswerTrData.COL_INTENT_TYPE].str.lower()
            self.bot_replies[BotIntentAnswerTrData.COL_CATEGORY] = self.bot_replies[BotIntentAnswerTrData.COL_CATEGORY].str.lower()
            # Create a unique intent ID
            self.bot_replies[BotIntentAnswerTrData.COL_INTENT_ID] = self.bot_replies[BotIntentAnswerTrData.COL_CATEGORY] +\
                '/' + self.bot_replies[BotIntentAnswerTrData.COL_INTENT]

            # Keep only these
            self.bot_replies = self.bot_replies[[
                BotIntentAnswerTrData.COL_BRAND,
                BotIntentAnswerTrData.COL_LANG,
                BotIntentAnswerTrData.COL_INTENT_TYPE,
                BotIntentAnswerTrData.COL_CATEGORY,
                BotIntentAnswerTrData.COL_WEIGHT,
                BotIntentAnswerTrData.COL_INTENT,
                BotIntentAnswerTrData.COL_INTENT_ID,
                BotIntentAnswerTrData.COL_ANSWER,
                BotIntentAnswerTrData.COL_TRAINING_DATA
            ]]

            # Make sure index is again 0,1,2,...
            self.bot_replies = self.bot_replies.reset_index(drop=True)

        except Exception as ex:
            raise(ex)

        # Extract the training data from a chunk into proper lines, also filter out non-user Intent Types
        self.training_data = None
        self.extract_training_data()

        # For Comm100 format
        self.intent_answer_trdata_comm100_format = None

        return

    def extract_training_data(self, do_comm100=True):
        # Ignore "system" and "regex" types
        df = self.bot_replies[
            (self.bot_replies[BotIntentAnswerTrData.COL_INTENT_TYPE] == BotIntentAnswerTrData.INTENT_TYPE_USER)
            | (self.bot_replies[BotIntentAnswerTrData.COL_INTENT_TYPE] == BotIntentAnswerTrData.INTENT_TYPE_INTERNAL)
            ]
        df = df.reset_index(drop=True)

        log.Log.log(str(self.__class__) + ' Extracting training data from data frame of size ' + str(df.shape[0]))

        self.training_data = pd.DataFrame()
        count = 0
        if do_comm100:
            self.intent_answer_trdata_comm100_format = pd.DataFrame()
        #
        # Loop by Intent
        #
        for i in range(0, df.shape[0], 1):
            brand = df[BotIntentAnswerTrData.COL_BRAND].loc[i]
            lang = df[BotIntentAnswerTrData.COL_LANG].loc[i]
            cat = df[BotIntentAnswerTrData.COL_CATEGORY].loc[i]
            intent_type = df[BotIntentAnswerTrData.COL_INTENT_TYPE].loc[i]
            intent = df[BotIntentAnswerTrData.COL_INTENT].loc[i]
            intent_id = df[BotIntentAnswerTrData.COL_INTENT_ID].loc[i]

            # Need to convert this whole chunk into individual training data
            td_chunk = df[BotIntentAnswerTrData.COL_TRAINING_DATA].loc[i]

            # Break it up into a list
            count = count + 1
            log.Log.log(str(count) + '. Extracting training data for intent [' + cat + '/' + intent + ']')
            td = td_chunk.split('\n')
            l = len(td)
            # Remove unnecessary spaces/tabs in front/behind
            for j in range(0, l, 1):
                td[j] = su.StringUtils.trim(td[j])
            log.Log.log(td)
            if len == 0:
                if intent_type == BotIntentAnswerTrData.INTENT_TYPE_USER:
                    raise Exception('No training data for intent [' + cat + '/' + intent + ']!!!!!')

            df_tmp = pd.DataFrame({
                BotIntentAnswerTrData.COL_BRAND:         l*[brand],
                BotIntentAnswerTrData.COL_LANG:          l*[lang],
                BotIntentAnswerTrData.COL_INTENT_TYPE:   l*[intent_type],
                BotIntentAnswerTrData.COL_CATEGORY:      l*[cat],
                BotIntentAnswerTrData.COL_INTENT:        l*[intent],
                BotIntentAnswerTrData.COL_INTENT_ID:     l*[intent_id],
                BotIntentAnswerTrData.COL_TEXT:          td
            })
            self.training_data = self.training_data.append(df_tmp, ignore_index=True)

            # For Comm100
            if do_comm100:
                response = df[BotIntentAnswerTrData.COL_ANSWER].loc[i]
                # Replace markups (Comm100 will remove)
                response = re.sub('<intentlink>', '(intentlink)', response)
                response = re.sub('</intentlink>', '(/intentlink)', response)
                response = re.sub('<form>', '(form)', response)
                response = re.sub('</form>', '(/form)', response)
                response = re.sub('<field>', '(field)', response)
                response = re.sub('</field>', '(/field)', response)
                response = re.sub('<webhook>', '(webhook)', response)
                response = re.sub('</webhook>', '(/webhook)', response)

                td_chunk_comm100 = '|'.join(td)

                df_tmp_comm100 = pd.DataFrame({
                    'Category Path':     [cat],
                    'Intent Name':       [intent],
                    'Visitor Questions': [td_chunk_comm100],
                    'Response':          [response]
                })
                self.intent_answer_trdata_comm100_format =\
                    self.intent_answer_trdata_comm100_format.append(df_tmp_comm100, ignore_index=True)

        log.Log.log('Extraction of training data DONE.')

        # Reset index so they are ordered 1,2,3... again
        self.training_data = self.training_data.reset_index(drop=True)
        self.intent_answer_trdata_comm100_format = self.intent_answer_trdata_comm100_format.reset_index(drop=True)

        log.Log.log('Writing training data to file [' + self.filepath_trdata + ']...')
        self.training_data.to_csv(path_or_buf=self.filepath_trdata, header=True, sep=',', index=False)
        log.Log.log('Successful!')

        # For Comm100
        if do_comm100:
            log.Log.log('Writing intent/answer/trdata of Comm100 format to file [' + self.filepath_intent_answer_trdata_comm100 + ']...')
            self.intent_answer_trdata_comm100_format.to_csv(
                path_or_buf=self.filepath_intent_answer_trdata_comm100, header=True, sep=',', index=False)
            log.Log.log('Successful!')

        return

    def get_intent_type(self, intent_id):
        df = self.bot_replies[self.bot_replies[BotIntentAnswerTrData.COL_INTENT_ID]==intent_id]
        if df.shape[0] == 0:
            return None
        df = df.reset_index(drop=True)
        return str(df[BotIntentAnswerTrData.COL_INTENT_TYPE].loc[0])

    def get_regex(self, intent_id):
        type = self.get_intent_type(intent_id=intent_id)
        if type is None:
            return None
        elif type == BotIntentAnswerTrData.INTENT_TYPE_REGEX:
            df = self.bot_replies[self.bot_replies[BotIntentAnswerTrData.COL_INTENT_ID] == intent_id]
            df = df.reset_index(drop=True)
            return str(df[BotIntentAnswerTrData.COL_TRAINING_DATA].loc[0])
        else:
            return None

    def get_regex_intent_ids(self):
        df = self.bot_replies[self.bot_replies[BotIntentAnswerTrData.COL_INTENT_TYPE] == BotIntentAnswerTrData.INTENT_TYPE_REGEX]
        if df.shape[0] == 0:
            return []
        else:
            return list(df[BotIntentAnswerTrData.COL_INTENT_ID])

    def get_intent_id(self, intent):
        df = self.bot_replies[self.bot_replies[BotIntentAnswerTrData.COL_INTENT] == intent]
        if df.shape[0] == 0:
            log.Log.log(str(self.__class__) + ' Could not find Intent ID for Intent [' + str(intent) + ']!!!')
            return None
        else:
            df = df.reset_index(drop=True)
            # Get the first one
            intent_id = df[BotIntentAnswerTrData.COL_INTENT_ID].loc[0]
            log.Log.log(str(self.__class__) + ' Found Intent ID [' + str(intent_id) + '] for Intent [' + str(intent) + ']')
            return(intent_id)

    #
    # Each intent may have as many replies as possible, we randomly choose 1 based on the weights given
    #
    def get_random_reply(self, intent_id):
        df = self.bot_replies[
            (self.bot_replies[BotIntentAnswerTrData.COL_BRAND]==self.brand)
            & (self.bot_replies[BotIntentAnswerTrData.COL_LANG]==self.lang)
            & (self.bot_replies[BotIntentAnswerTrData.COL_INTENT_ID]==intent_id)
        ]
        df = df.reset_index(drop=True)

        if df.shape[0] == 0:
            log.Log.log('Warning! For (brand, lang)=(' +
                        str(self.brand) + ', ' + str(self.lang) +
                        '), intent ID [' + str(intent_id) + '] Not found.')
            return None

        cum_weight = df[BotIntentAnswerTrData.COL_WEIGHT].cumsum()
        total_weight = sum(df[BotIntentAnswerTrData.COL_WEIGHT])

        # Choose a random number between [0,total_weight), 0 inclusive, total_weight exclusive
        randomno = np.random.randint(low=0, high=total_weight)
        df_gt = df[cum_weight>randomno]
        # Make sure indexes are ordered 0,1,2... again
        df_gt = df_gt.reset_index(drop=True)
        # The reply we randomly chose is the first cumulative weight greater than the random number
        reply = df_gt[BotIntentAnswerTrData.COL_ANSWER].loc[0]

        # Depending on answer type, whether to call webhook, parse markups, etc.
        #anstype = df_gt[BotReply.COL_ANSWER_TYPE].loc[0]

        # TODO Support webhook answer type
        #if anstype == BotReply.ANSWER_TYPE_WEBHOOK:
        #    raise Exception('Webhook answer type not yet supported!')

        return reply


def demo():
    dirpath = '/Users/mark.tan/svn/yuna/app.data/chatbot/traindata'
    postfix_iat = 'chatbot.intent-answer-trdata'
    postfix_trd = 'chatbot.trainingdata'
    br = BotIntentAnswerTrData(
        dirpath=dirpath,
        postfix_intent_answer_trdata=postfix_iat,
        postfix_trdata=postfix_trd,
        brand='fun88',
        lang='cn')
    #log.Log.log(br.bot_replies)
    #log.Log.log(br.training_data)

    print(br.get_intent_type(intent_id='payment/存提款单号'))
    print(br.get_regex(intent_id='payment/存提款单号'))
    print(br.get_intent_type(intent_id='alsdkjf'))
    print(br.get_regex(intent_id='alsdkjf'))

    print(br.get_regex_intent_ids())

    print(br.get_intent_id(intent='无法充值'))

    v = []
    for i in range(0,100,1):
        msg = br.get_random_reply(intent_id=BotIntentAnswerTrData.INTENT_WELCOME)
        #msg = br.get_random_reply(intent_id='common/必威常见问题')
        v = v + [msg]
        #print(msg)

    cv = collections.Counter(v)
    cv.most_common()
    print(cv)

if __name__ == '__main__':
    demo()