# -*- coding: utf-8 -*-

import pandas as pd
import nwae.lib.lang.LangFeatures as lf
import nwae.lib.lang.nlp.LatinEquivalentForm as latinEqForm
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor
from nwae.lib.lang.preprocessing.TxtPreprocessor import TxtPreprocessor
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
from nwae.lib.lang.nlp.daehua.DaehuaTrainDataModel import DaehuaTrainDataModel
import nwae.lib.math.ml.Trainer as nwaeTrainer


#
# Preprocessing of raw text training data into desired form.
#
class TrDataPreprocessor:

    TRDATA_ID_INTENT_NAME = -1
    TRDATA_ID_LATIN_FORM = -2

    def __init__(
            self,
            model_identifier,
            language,
            # Training data in pandas DataFrame format with at least 3 columns
            #   DaehuaTrainDataModel.COL_TDATA_INTENT_ID
            #   DaehuaTrainDataModel.COL_TDATA_INTENT_NAME
            #   DaehuaTrainDataModel.COL_TDATA_TEXT
            #   DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED
            #   DaehuaTrainDataModel.COL_TDATA_TRAINING_DATA_ID
            df_training_data,
            dirpath_wordlist,
            postfix_wordlist,
            dirpath_app_wordlist,
            postfix_app_wordlist,
            dirpath_synonymlist,
            postfix_synonymlist,
            # Do word processing for all sentences, when word/synonym list changes
            reprocess_all_text = False
    ):
        self.model_identifier = model_identifier
        self.language = language
        self.df_training_data = df_training_data
        self.dirpath_wordlist = dirpath_wordlist
        self.postfix_wordlist = postfix_wordlist
        self.dirpath_app_wordlist = dirpath_app_wordlist
        self.postfix_app_wordlist = postfix_app_wordlist
        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist

        self.reprocess_all_text = reprocess_all_text
        # The caller might want to update his DB
        self.list_of_rows_with_changed_processed_text = []

        lfobj = lf.LangFeatures()
        self.lang_have_verb_conj = lfobj.have_verb_conjugation(lang=self.language)
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Language "' + str(self.language)
            + '" have verb conjugation = ' + str(self.lang_have_verb_conj) + '.'
        )

        self.txt_preprocessor = TxtPreprocessor(
            identifier_string      = self.model_identifier,
            # Don't need directory path for model, as we will not do spelling correction
            dir_path_model         = None,
            # Don't need features/vocabulary list from model
            model_features_list    = None,
            lang                   = self.language,
            dirpath_synonymlist    = self.dirpath_synonymlist,
            postfix_synonymlist    = self.postfix_synonymlist,
            dir_wordlist           = self.dirpath_wordlist,
            postfix_wordlist       = self.postfix_wordlist,
            dir_wordlist_app       = self.dirpath_app_wordlist,
            postfix_wordlist_app   = self.postfix_app_wordlist,
            do_spelling_correction = False,
            do_word_stemming       = self.lang_have_verb_conj,
            do_profiling           = False
        )

        # We will convert the dataframe into our nwae format
        self.nwae_training_data = None

        return

    def __get_row_to_append_to_training_data(
            self,
            intent_id,
            intent_name,
            text,
            text_id,
            processed_text
    ):
        return {
            DaehuaTrainDataModel.COL_TDATA_INTENT_ID: intent_id,
            DaehuaTrainDataModel.COL_TDATA_INTENT_NAME: intent_name,
            DaehuaTrainDataModel.COL_TDATA_TEXT: text,
            DaehuaTrainDataModel.COL_TDATA_TRAINING_DATA_ID: text_id,
            DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED: processed_text
        }

    #
    # Our interface to external objects so that they can start this lengthy process in the background
    #
    def go(self):
        self.add_intent_name_to_training_data()
        self.process_text_training_data()
        self.add_latin_form_to_training_data()
        self.nwae_training_data = nwaeTrainer.Trainer.convert_to_training_data_model_type(
            td = self.df_training_data,
            lang = self.language
        )
        return self.nwae_training_data

    def add_intent_name_to_training_data(self):
        #
        # We need to add intent name into the training data also
        #
        df_intent_id_name = pd.DataFrame(
            {
                DaehuaTrainDataModel.COL_TDATA_INTENT_ID:
                    self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_ID],
                DaehuaTrainDataModel.COL_TDATA_INTENT_NAME:
                    self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME]
            })
        # Make unique by dropping duplicate intent IDs
        df_intent_id_name.drop_duplicates(inplace=True)

        for idx in df_intent_id_name.index:
            intId = df_intent_id_name[DaehuaTrainDataModel.COL_TDATA_INTENT_ID].loc[idx]
            try:
                int_name = str(df_intent_id_name[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME].loc[idx])

                # Arguments be a list form, otherwise will not be able to create this DataFrame
                row_to_append = pd.DataFrame(
                    data = self.__get_row_to_append_to_training_data(
                        intent_id      = [intId],
                        intent_name    = [int_name],
                        text           = [int_name],
                        text_id        = [TrDataPreprocessor.TRDATA_ID_INTENT_NAME],
                        # Make sure to write back this value with processed text
                        processed_text = [None]
                ))

                #
                # We are appending to a dataframe that might have different columns ordering
                # So we make sure they are in the same order, to avoid all the sort=False/True
                # warning messages by pandas due to required join() operation.
                # If in same order, then we avoid the join().
                #
                self.df_training_data = self.df_training_data.append(
                    row_to_append,
                    sort = True
                )
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Appended intent name "' + str(int_name) + '" with intent ID ' + str(intId)
                    + ' to list of training data. Row appended = ' + str(row_to_append)
                )
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Could not append to dataframe or could not get intent name for intent ID ' \
                    + str(intId) + '. Exception ' + str(ex)
                log.Log.warning(errmsg)
                raise Exception(errmsg)

        self.__process_training_data_index()

        #
        # Now we decode all Daehua Training Language Model
        # TODO Remove all this when Daehua Model is encoded in separate columns and not under Intent Name
        #
        # dh_obj = DaehuaTrainDataModel(
        #     daehua_training_data = self.df_training_data
        # )
        # # This will clean 2 columns: Intent Names & Txt
        # self.df_training_data = dh_obj.clean_daehua_training_data()
        # log.Log.info(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Done cleaning training data from daehua model variables.'
        # )
        #
        # log.Log.debug(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': After cleaning daehua variables, 10 Lines training data:\n\r'
        #     + str(self.df_training_data.columns) + '\n\r'
        #     + str(self.df_training_data[1:10].values)
        #     + '\n\r: Shape: ' + str(self.df_training_data.shape)
        # )

        return self.df_training_data

    def __process_training_data_index(self):
        # Sort by Intent ID and reset index
        self.df_training_data = self.df_training_data.sort_values(
            [DaehuaTrainDataModel.COL_TDATA_INTENT_ID],
            ascending=True
        )
        self.df_training_data = self.df_training_data.reset_index(drop=True)

        # Now derive the training data index
        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Assigning numbers to training data based on intent...'
        )
        # Add intent index
        self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_INDEX] =\
            [0]*self.df_training_data.shape[0]
        prev_cat_int = ''
        prev_cat_int_index = 0
        for i in range(0, self.df_training_data.shape[0], 1):
            cur_cat_int = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_ID].loc[i]
            if cur_cat_int != prev_cat_int:
                prev_cat_int = cur_cat_int
                prev_cat_int_index = 0
            prev_cat_int_index = prev_cat_int_index + 1
            self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_INDEX].at[i] = prev_cat_int_index

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': After process training data index, 10 Lines training data:\n\r'
            + str(self.df_training_data.columns) + '\n\r'
            + str(self.df_training_data[1:10].values)
            + '\n\r: Shape: ' + str(self.df_training_data.shape)
        )
        return


    #
    # Segment text if necessary, stem if necessary for languages like English
    # This can happen when there is new training text in the DB, and the BO just write NULL
    # to the segmented text column.
    # TODO Do we also need to normalize text?
    #
    def process_text_training_data(
            self,
    ):
        # The algorithm to segment words works as follows:
        #   If segmented text returned from DB is None or shorter than text, we will process the text.
        #   However if the flag self.reprocess_all_text == True, we segment no matter what.

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': START SEGMENT & STEM DB TRAINING DATA, FORCE RESEGMENT ALL = '
            + str(self.reprocess_all_text)
        )

        td_total_rows = self.df_training_data.shape[0]
        count = 0

        for idx_row in self.df_training_data.index:
            count = count + 1
            text_from_db = str(self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TEXT].loc[idx_row])
            text_processed_from_db = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED].loc[idx_row]
            intent_td_id = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TRAINING_DATA_ID].loc[idx_row]
            intent_id = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_ID].loc[idx_row]
            intent_name = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME].loc[idx_row]

            log.Log.debugdebug(
                'Processing index row "' + str(idx_row) + '" '
                + str(self.df_training_data.loc[idx_row]) + '"'
            )

            if type(text_from_db) is not str:
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Text from DB "' + str(text_from_db) + '" not string type.'
                )
                text_from_db = str(text_from_db)
            if (text_processed_from_db is None) or (str(text_processed_from_db).lower() == 'none'):
                text_processed_from_db = ''

            #
            # Sanity check only. Should not happen since after every training data update,
            # NULL would be written back to the TextSegmented column.
            # Because we don't want to reprocess all text which takes time, so we guess first
            #
            is_likely_processed_text_changed = len(text_processed_from_db) < len(text_from_db)
            # If a language has verb conjugation, we cannot just compare length as the original text could be longer
            if self.lang_have_verb_conj:
                # So we just hardcode
                is_likely_processed_text_changed = len(text_processed_from_db) <= 8

            if is_likely_processed_text_changed:
                if (intent_td_id is not None) and (intent_td_id > 0):
                    # Warn only if it is not our own inserted data
                    log.Log.warning(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Text "' + str(text_from_db)
                        + '" likely has incorrect segmentation "' + str(text_processed_from_db) + '".'
                    )

            #
            # We only reprocess the text if there is some likelihood of change
            #
            if self.reprocess_all_text or is_likely_processed_text_changed:
                processed_text_str = self.txt_preprocessor.process_text(
                    inputtext = text_from_db,
                    return_as_string = True
                )
                log.Log.debug(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Text "' + str(text_from_db) + '" processed text "' + str(processed_text_str) + '".'
                )

                is_text_processed_changed = not (text_processed_from_db == processed_text_str)
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': No ' + str(count) + ' of ' + str(td_total_rows)
                    + ': Tr Data ID "' + str(intent_td_id)
                    + '". Force segment = ' + str(self.reprocess_all_text)
                    + '\n\r   Text "' + str(text_from_db) + '". Processed to "' + str(processed_text_str) + '"'
                    + ', changed = ' + str(is_text_processed_changed)
                )

                # Training ID 0 are those we inserted ourselves so no need to update anything
                if is_text_processed_changed:
                    # Update the column
                    self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED].at[idx_row] = \
                        processed_text_str

                    # For intent name we inserted, no need to warn
                    if (intent_td_id is not None) and (intent_td_id > 0):
                        log.Log.warning(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Processed text different. Text "' + str(text_from_db)
                            + '\n\r   new processed text "' + str(processed_text_str) + '"'
                            + '\n\r   old processed text "' + str(text_processed_from_db) + '"'
                        )

                        row_changed = self.__get_row_to_append_to_training_data(
                            intent_id      = intent_id,
                            intent_name    = intent_name,
                            text           = text_from_db,
                            text_id        = intent_td_id,
                            processed_text = processed_text_str
                        )
                        self.list_of_rows_with_changed_processed_text.append(row_changed)
                        log.Log.important(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Appended changed row: ' + str(row_changed)
                        )
                    else:
                        log.Log.important(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Processed text ' + str(count) + ' ok "' + str(processed_text_str)
                            + '" from "' + str(text_from_db) + '"'
                        )
            else:
                log.Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Training data ID ' + str(intent_td_id)
                    + ': No ' + str(count) + ' of ' + str(td_total_rows)
                    + ': Nothing to do, OK segmented/processed from DB "' + str(text_processed_from_db) + '"'
                )
        return

    #
    # For some languages like Vietnamese, there is very high tendency for a native speaker
    # to type in incorrect casual latin, instead of pure Vietnamese.
    # Thus for every sentence, we add the latin converted form also.
    # But this needs to be done after segmentation & all text processing,
    # as the word segmenter will not know how to segment the incorrect casual latin form.
    #
    def add_latin_form_to_training_data(
            self
    ):
        if not latinEqForm.LatinEquivalentForm.have_latin_equivalent_form(lang = self.language):
            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': For "' + str(self.model_identifier) + '", language "' + str(self.language)
                + '", nothing to do for latin equivalent form.'
            )
            return

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': For "' + str(self.model_identifier) + '", language "' + str(self.language)
            + '", adding to training data, the latin equivalent form.'
        )
        for idx in self.df_training_data.index:
            text = str(self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TEXT].loc[idx])
            text_processed = str(self.df_training_data[DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED].loc[idx])
            #
            # Process the sentence, word by word
            #
            word_sep = BasicPreprocessor.get_word_separator(lang=self.language)
            latin_form_sentence_arr = []
            for word in text_processed.split(sep=word_sep):
                word_latin = latinEqForm.LatinEquivalentForm.get_latin_equivalent_form(
                    lang = self.language,
                    word = word
                )
                latin_form_sentence_arr.append(word_latin)
            latin_form_sentence_txt = word_sep.join(latin_form_sentence_arr)
            if latin_form_sentence_txt == text_processed:
                continue

            log.Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Processing latin equivalent form "' + str(latin_form_sentence_txt)
                + '" for sentence "' + str(text_processed) + '".'
            )
            int_id = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_ID].loc[idx]
            int_name = self.df_training_data[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME].loc[idx]
            row_to_append = None
            try:
                # Arguments be a list form, otherwise will not be able to create this DataFrame
                row_to_append = pd.DataFrame(
                    data = self.__get_row_to_append_to_training_data(
                        intent_id      = [int_id],
                        intent_name    = [int_name],
                        text           = [text],
                        text_id        = [TrDataPreprocessor.TRDATA_ID_LATIN_FORM],
                        processed_text = [latin_form_sentence_txt]
                    ))
                #
                # We are appending to a dataframe that might have different columns ordering
                # So we make sure they are in the same order, to avoid all the sort=False/True
                # warning messages by pandas due to required join() operation.
                # If in same order, then we avoid the join().
                #
                self.df_training_data = self.df_training_data.append(
                    row_to_append,
                    sort = True
                )
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Appended latin equivalent form "' + str(latin_form_sentence_txt)
                    + '" with intent ID ' + str(int_id)
                    + ' to list of training data. Row appended = ' + str(row_to_append)
                )
            except Exception as ex:
                errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Could not append row ' + str(row_to_append) + ' to dataframe for intent ID ' \
                    + str(int_id) + '. Exception ' + str(ex)
                log.Log.warning(errmsg)
                raise Exception(errmsg)
        self.__process_training_data_index()
        return


if __name__ == '__main__':
    from nwae.lib.lang.preprocessing.ut.UtTrDataPreprocessor import UtTrDataPreprocessor
    from nwae.config.Config import Config
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = Config,
        default_config_file = '/usr/local/git/nwae/nwae/app.data/config/local.nwae.cf'
    )
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    UtTrDataPreprocessor(config).run_unit_test()