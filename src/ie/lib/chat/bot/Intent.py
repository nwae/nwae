# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
import threading
import ie.lib.lang.model.FeatureVector as fv
import ie.lib.chat.classification.training.RefFeatureVec as reffv
import ie.lib.lang.nlp.SynonymList as sl
import ie.lib.util.Log as log


#
# Intent AI/NLP Engine
# TODO: Need to rename this class la
#
class Intent:

    SEARCH_TOPX_RFV = 10
    DEFAULT_SCORE_MIN_THRESHOLD = 10
    # Weight given to distance to RFV & closest sample, to determine the distance to a command
    WEIGHT_RFV = 0.5
    WEIGHT_SAMPLE = 1 - WEIGHT_RFV

    # Column names for intent detection data frame
    COL_TEXT_NORMALIZED = 'TextNormalized'
    COL_COMMAND = reffv.RefFeatureVector.COL_COMMAND
    COL_DISTANCE_TO_RFV = 'DistToRfv'
    COL_DISTANCE_CLOSEST_SAMPLE = 'DistToSampleClosest'
    COL_DISTANCE_FURTHEST = reffv.RefFeatureVector.COL_DISTANCE_TO_RFV_FURTHEST
    COL_MATCH = 'Match'
    COL_SCORE = 'Score'
    COL_SCORE_CONFIDENCE_LEVEL = 'ScoreConfLevel'

    # From rescoring training data, we find that
    #    5% quartile score  = 55
    #    25% quartile score = 65
    #    50% quartile score = 70
    #    75% quartile score = 75
    #    95% quartile score = 85
    # Using the above information, we set
    CONFIDENCE_LEVEL_5_SCORE = 75
    CONFIDENCE_LEVEL_4_SCORE = 65
    CONFIDENCE_LEVEL_3_SCORE = 55
    # For confidence level 0-2, we run the bot against non-related data and we found
    #    99% quartile score = 32
    #    95% quartile score = 30
    #    75% quartile score = 20
    CONFIDENCE_LEVEL_2_SCORE = 40   # Means <1% of non-related data will go above it
    CONFIDENCE_LEVEL_1_SCORE = 20   # This means 25% of non-related data will go above it


    def __init__(self,
                 lang,
                 brand,
                 dir_rfv_commands,
                 dirpath_synonymlist):
        self.lang = lang.lower()
        self.brand = brand.lower()
        self.dir_rfv_commands = dir_rfv_commands

        self.df_idf = None
        self.df_rfv = None
        self.df_rfv_dist_furthest = None
        # Used to zoom into an intent/command group and compare against exact training data in that group
        self.df_fv_training_data = None
        # Used to finally confirm if it is indeed the intent by matching top keywords in the intent category
        self.df_intent_tf = None

        self.synonymlist = sl.SynonymList(lang=lang,
                                          dirpath_synonymlist=dirpath_synonymlist,
                                          postfix_synonymlist='.synonymlist.txt')
        self.synonymlist.load_synonymlist(verbose=1)

        # Thread safe
        self.mutex = threading.Lock()
        return

    def load_rfv_commands_from_file(self):
        fpath_idf = self.dir_rfv_commands + '/' + self.lang + '.' + self.brand + '.' + 'chatbot.words.idf.csv'
        self.df_idf = pd.read_csv(filepath_or_buffer=fpath_idf, sep=',', index_col=0)

        fpath_rfv = self.dir_rfv_commands + '/' + self.lang + '.' + self.brand + '.' + 'chatbot.commands.rfv.csv'
        self.df_rfv = pd.read_csv(filepath_or_buffer=fpath_rfv, sep=',', index_col=0)

        fpath_rfv_dist = self.dir_rfv_commands + '/' + self.lang + '.' + self.brand + '.' + 'chatbot.commands.rfv.distance.csv'
        self.df_rfv_dist_furthest = pd.read_csv(filepath_or_buffer=fpath_rfv_dist, sep=',', index_col=0)

        fpath_fv_all = self.dir_rfv_commands + '/' + self.lang + '.' + self.brand + '.' + 'chatbot.fv.all.csv'
        self.df_fv_training_data = pd.read_csv(filepath_or_buffer=fpath_fv_all, sep=',', index_col=0)

        fpath_intent_tf = self.dir_rfv_commands + '/' + self.lang + '.' + self.brand + '.' + 'chatbot.commands.words.tf.csv'
        self.df_intent_tf = pd.read_csv(filepath_or_buffer=fpath_intent_tf, sep=',', index_col=0)

        # Check RFV is normalized
        for com in list(self.df_rfv.index):
            rfv = np.matrix(self.df_rfv.loc[com].values)
            dist = np.sum(np.multiply(rfv,rfv))**0.5
            if abs(dist-1) > 0.000001:
                log.Log.log('Warning: RFV for command [' + str(com) + '] not 1, ' + str(dist))

        for com in list(self.df_fv_training_data.index):
            fv = np.matrix(self.df_fv_training_data.loc[com].values)
            dist = np.sum(np.multiply(fv,fv))**0.5
            if abs(dist-1) > 0.000001:
                log.Log.log('Warning: FV for command [' + str(com) + '] not 1, ' + str(dist))
                log.Log.log(fv)
                raise Exception('FV error')

        return

    #
    # This is the external interface for the CRM/etc. bot to call to get text intent/command.
    # Text is assumed to be already word segmented by ' '
    # TODO: Can't use ' ' as word boundary for Vietnamese
    # TODO: "No Match" also need to return
    #
    def get_text_class(self,
                       text_segmented,
                       weigh_idf,
                       normalized=True,
                       top=SEARCH_TOPX_RFV,
                       return_match_results_only=True,
                       score_min_threshold=DEFAULT_SCORE_MIN_THRESHOLD,
                       verbose=0):

        # Replace words with root words
        text_normalized = self.synonymlist.normalize_text(text_segmented=text_segmented, verbose=verbose)
        text_normalized = text_normalized.lower()

        keywords = list(self.df_rfv.columns)

        model_fv = fv.FeatureVector()
        model_fv.set_freq_feature_vector_template(list_symbols=keywords)
        if weigh_idf:
            model_fv.set_feature_weights(fw=list(self.df_idf['IDF']))

        # Get feature vector of text
        df_fv = model_fv.get_freq_feature_vector(text_normalized)
        # This creates a single row matrix that needs to be transposed before matrix multiplications
        fv_text = np.matrix(df_fv['Frequency'].values)
        fv_text_normalized = np.matrix(df_fv['FrequencyNormalized'].values)

        if normalized:
            fv_text = fv_text_normalized

        #fv_text_feature_presence = (fv_text > 0) * 1

        #
        # Step 1:
        #    Pre-filter and remove intent RFVs which have no common features.
        #    This speeds up processing by a lot, more than half.
        # By right, if we don't do a 2 step processing, we can directly compare with the entire training
        # data set FV, which will not be slow due to matrix operations.
        # However it is mathematically inelegant and maybe not extensible in the future as training data grows.
        #
        rfv_matrix = np.matrix(self.df_rfv.values)
        # Multiply matrices to see which rows are zero, returns a single column matrix
        rfv_text_mul = rfv_matrix * fv_text.transpose()
        # Get non-zero intents, those that have at least a single intersecting feature
        non_zero_intents = rfv_text_mul.transpose() > 0
        non_zero_intents = non_zero_intents.tolist()
        # Only select the RFVs that have common features with the text
        df_rfv_nonzero = self.df_rfv.loc[non_zero_intents[0]]

        # If no matches
        if df_rfv_nonzero.shape[0] == 0:
            if verbose >= 1:
                log.Log.log('No common features with RFV!!')
            return None

        # Create a matrix of similar rows (fv_text)
        text_matrix = np.repeat(a=fv_text, repeats=df_rfv_nonzero.shape[0], axis=0)
        rfv_nonzero_matrix = np.matrix(df_rfv_nonzero)
        dif_rfv_text = rfv_nonzero_matrix - text_matrix
        # Square every element in the matrix
        dif_rfv_text2 = np.power(dif_rfv_text, 2)
        # Sum every row to create a single column matrix
        dif_rfv_text2_sum = dif_rfv_text2.sum(axis=1)
        # Take the square root of every element in the single column matrix as distance
        distance_rfv_text = np.power(dif_rfv_text2_sum, 0.5)
        # Convert to a single row matrix
        distance_rfv_text = distance_rfv_text.transpose()
        distance_rfv_text = distance_rfv_text.tolist()

        #
        # Keep top X matches
        #   1) Distance criteria of given text FV to all intent RFV
        #   2) Presence of features, how many matches in proportion
        #
        close_commands = list(df_rfv_nonzero.index)

        # Furthest distances to RFV in training data
        condition = self.df_rfv_dist_furthest[Intent.COL_COMMAND].isin(close_commands)
        df_distance_furthest = self.df_rfv_dist_furthest[condition]
        close_commands_distance_furthest = df_distance_furthest[Intent.COL_DISTANCE_FURTHEST].tolist()

        tmp_len = len(close_commands)
        df_dist_to_classes = pd.DataFrame(data={Intent.COL_TEXT_NORMALIZED: [text_normalized]*tmp_len,
                                                Intent.COL_COMMAND: close_commands,
                                                Intent.COL_DISTANCE_TO_RFV: distance_rfv_text[0],
                                                Intent.COL_DISTANCE_CLOSEST_SAMPLE: [99999999]*tmp_len,
                                                Intent.COL_DISTANCE_FURTHEST: close_commands_distance_furthest,
                                                Intent.COL_MATCH: [0]*tmp_len,
                                                Intent.COL_SCORE: [0]*tmp_len,
                                                Intent.COL_SCORE_CONFIDENCE_LEVEL: [0]*tmp_len
                                                })

        # Sort distance ascending
        df_dist_to_classes = df_dist_to_classes.sort_values(by=[Intent.COL_DISTANCE_TO_RFV], ascending=True)
        df_dist_to_classes = df_dist_to_classes.reset_index(drop=True)

        #
        # Zooming into the intent category
        # We do more detailed calculations now, to see which of the top intents is the best match.
        #
        # Only do top X
        top_intents = list( df_dist_to_classes[Intent.COL_COMMAND].loc[0:(top-1)].values )

        if verbose >= 2:
            log.Log.log('Top intents: [' + str(top_intents) + ']')
            log.Log.log(df_dist_to_classes.loc[0:(top-1)])

        # Doing only for top intents save time, speeds up the process by a lot
        for intent in top_intents:
            intent_index = df_dist_to_classes[df_dist_to_classes[Intent.COL_COMMAND] == intent].index

            # Use training data FVs (if text contains '?', '+',... need to replace with '[?]',... otherwise regex will fail
            intent_re = intent
            for specchar in ['?','+','*','(',')']:
                intent_re = intent_re.replace(specchar,'['+specchar+']')

            filter_intent = intent_re + '.[0-9]+.*'
            df_fv_td_intent = self.df_fv_training_data.filter(regex=filter_intent, axis=0)
            if df_fv_td_intent.shape[0] == 0:
                log.Log.log('Warning! For intent [' + intent + '] - No Training Data Filtered!')
                #log.Log.log(self.df_fv_training_data.filter(regex=intent.replace('?','[?]')+'..*', axis=0))
                continue

            # Create a matrix of similar rows (fv_text)
            tmp_text_matrix = np.repeat(a=fv_text, repeats=df_fv_td_intent.shape[0], axis=0)
            fv_td_intent_matrix = np.matrix(df_fv_td_intent)
            dif_fv_text = fv_td_intent_matrix - tmp_text_matrix
            # Square every element in the matrix
            dif_fv_text2 = np.power(dif_fv_text, 2)
            # Sum every row to create a single column matrix
            dif_fv_text2_sum = dif_fv_text2.sum(axis=1)
            # Take the square root of every element in the single column matrix as distance
            distance_fv_text = np.power(dif_fv_text2_sum, 0.5)
            # Convert to a single row matrix
            distance_fv_text = distance_fv_text.transpose()
            distance_fv_text = distance_fv_text.tolist()

            distance_sample_nearest = min(distance_fv_text[0])
            df_dist_to_classes.at[intent_index, Intent.COL_DISTANCE_CLOSEST_SAMPLE] = round(distance_sample_nearest, 5)
            if verbose >= 2:
                log.Log.log('   Closest distance to intent [' + intent + '] sample = ' + str(round(distance_sample_nearest, 5)))

        # Just in case there is only 1 entry (unlikely), we multiply by 1.1 to avoid a 0 score
        distance_threshold = 1.1 * max(df_dist_to_classes[Intent.COL_DISTANCE_TO_RFV])
        # Minimum of the threshold is 1.0
        distance_threshold = max(1, distance_threshold)
        if verbose >= 2:
            log.Log.log('Using distance threshold ' + str(distance_threshold))

        #
        # Now for the most important calculations: MATCH & SCORE
        #
        for intent in list(df_dist_to_classes[Intent.COL_COMMAND]):
            intent_index = df_dist_to_classes[df_dist_to_classes[Intent.COL_COMMAND]==intent].index

            dist_to_rfv = df_dist_to_classes[Intent.COL_DISTANCE_TO_RFV].loc[intent_index]
            dist_to_rfv = float(dist_to_rfv)
            dist_to_closest_sample = df_dist_to_classes[Intent.COL_DISTANCE_CLOSEST_SAMPLE].loc[intent_index]
            dist_to_closest_sample = float(dist_to_closest_sample)
            #
            # TODO: Instead of using the minimum of both, I think better to use a weighted average.
            # TODO: This will avoid training outlier problems, or downright wrong training data.
            #
            #dist_min = min(dist_to_rfv, dist_to_closest_sample)
            dist_min = dist_to_rfv*Intent.WEIGHT_RFV + dist_to_closest_sample*Intent.WEIGHT_SAMPLE

            #distance_threshold = df_dist_to_classes[Intent.COL_DISTANCE_FURTHEST].loc[intent_index]
            #distance_threshold = float(distance_threshold)

            #
            # A match to an intent is determined using the following factors:
            # - Distance (either distance to RFV or to closest sample) is less than threshold set
            #
            intent_match = (dist_min <= distance_threshold) * 1

            df_dist_to_classes.at[intent_index, Intent.COL_MATCH] = intent_match

            # Calculate our measure of "Score", which is somewhat arbitrary
            diff = max(distance_threshold - dist_min, 0)
            range_allowed = max(distance_threshold, 0.00001)
            diff_normalized = min(1, max(0, (diff / range_allowed)))

            #
            # FINALLY THE SCORE:
            # We want to create a curve that is y=1 when x=0, and y=0 when x=1, but decreases very slowly.
            # Thus we use cos(x)^k function, where k<1
            score = round(100 * diff_normalized, 2) * intent_match
                #round(100 * (math.cos(math.pi*diff_normalized/2)**0.2) * diff_normalized, 2)
            df_dist_to_classes.at[intent_index, Intent.COL_SCORE] = score
            # Maximum confidence level is 5, minimum 0
            df_dist_to_classes.at[intent_index, Intent.COL_SCORE_CONFIDENCE_LEVEL] = \
                (score >= Intent.CONFIDENCE_LEVEL_1_SCORE)*1 +\
                (score >= Intent.CONFIDENCE_LEVEL_2_SCORE)*1 +\
                (score >= Intent.CONFIDENCE_LEVEL_3_SCORE)*1 +\
                (score >= Intent.CONFIDENCE_LEVEL_4_SCORE)*1 +\
                (score >= Intent.CONFIDENCE_LEVEL_5_SCORE)*1

        # Sort score descending
        df_dist_to_classes = df_dist_to_classes.sort_values(by=[Intent.COL_SCORE], ascending=False)
        # Extract only matches
        if return_match_results_only:
            df_dist_to_classes = df_dist_to_classes[df_dist_to_classes[Intent.COL_MATCH]==1]
        # Extract only scores greater than threshold
        df_dist_to_classes = df_dist_to_classes[df_dist_to_classes[Intent.COL_SCORE] >= score_min_threshold]

        # Reset indexes
        df_dist_to_classes = df_dist_to_classes.reset_index(drop=True)

        if df_dist_to_classes.shape[0] >= 1:
            return df_dist_to_classes[0:min(top, df_dist_to_classes.shape[0])].copy()
        else:
            return None

