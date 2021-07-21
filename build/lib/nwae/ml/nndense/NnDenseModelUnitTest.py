# -*- coding: utf-8 -*-

import nwae.ml.TrainingDataModel as tdm
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
import numpy as np
import nwae.utils.UnitTest as ut
from nwae.ml.text.TxtTransform import TxtTransform
from nwae.ml.nndense.NnDenseModel import NnDenseModel
from nwae.ml.networkdesign.NetworkDesign import NetworkDesign
from nwae.ml.trainer.TextTrainer import TextTrainer
from nwae.ml.modelhelper.TextModelHelper import TextModelHelper

try:
    from keras.utils import to_categorical
except Exception as ex_keras:
    Log.warning(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Exception importing Keras modules: ' + str(ex_keras)
    )


class NnDenseModelUnitTest:

    THRESHOLD_PASS_PCT = 75.0
    THRESHOLD_PASS_PCT_LOW = 60.0

    TRAIN_EPOCHS = 100

    def __init__(self, ut_params):
        self.ut_params = ut_params

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)
        # Demonstrating a network architecture requiring convertion to categorical the labels
        res_final.update(other_res_obj=self.test_math_function())
        # Demonstrating a network architecture with Flatten layer thus does not require categorical labels
        res_final.update(other_res_obj=self.test_text_classification())
        res_final.update(other_res_obj=self.test_text_classification_from_sample_data())
        return res_final

    def __check_result_of_test(
            self,
            # numpy ndarray type, input data
            data,
            labels,
            # If given, we also check against this
            labels_original,
            model_obj,
            res_obj,
            # Expected accuracy in %
            exp_acc,
    ):
        n_rows = data.shape[0]
        # Predict all at once
        top_match_bulk = model_obj.predict_classes(x=data).predicted_classes_coded

        # Compare some data, do it one by one
        count_correct_bulk = 0
        count_correct = 0
        count_correct_original = 0
        for i in range(data.shape[0]):
            data_i = np.array([data[i]])
            label_i = labels[i]
            if labels_original is not None:
                label_original_i = labels_original[i]
            else:
                label_original_i = None
            pred_res = model_obj.predict_class(
                x = data_i
            )
            top_matches_coded = pred_res.predicted_classes_coded
            top_matches_original_labels = pred_res.predicted_classes
            top_match = top_matches_coded[0]
            if top_match != label_i:
                Log.debug(
                    str(i) + '. ' + str(data_i) + ': Label=' + str(label_i)
                    + ', wrongly predicted=' + str(top_match)
                )
            if top_matches_original_labels is not None:
                top_match_original_label = top_matches_original_labels[0]
                if label_original_i is not None:
                    if top_match_original_label != label_original_i:
                        Log.debug(
                            str(i) + '. ' + str(data_i) + ': Label=' + str(label_original_i)
                            + ', wrongly predicted=' + str(top_match_original_label)
                        )
                    count_correct_original += 1 * (top_match_original_label == label_original_i)

            count_correct      += 1 * (top_match == label_i)
            count_correct_bulk += 1 * (top_match_bulk[i] == label_i)

        accuracy_manual_check      = round(100 * count_correct / data.shape[0], 2)
        accuracy_manual_check_bulk = round(100 * count_correct_bulk / data.shape[0], 2)
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Accuracy manual = ' + str(accuracy_manual_check) + '%.'
        )
        res_obj.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = (accuracy_manual_check == exp_acc),
            expected = True,
            test_comment = ': Test manual predict accuracy: ' + str(accuracy_manual_check)
                           + ' on dataset with ' + str(n_rows) + ' rows data'
        ))
        res_obj.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = (accuracy_manual_check_bulk == exp_acc),
            expected = True,
            test_comment = ': Test bulk predict accuracy: ' + str(accuracy_manual_check_bulk)
                           + ' on dataset with ' + str(n_rows) + ' rows data'
        ))
        if labels_original is not None:
            accuracy_manual_check_original = round(100 * count_correct_original / data.shape[0], 2)
            res_obj.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = (accuracy_manual_check_original == exp_acc),
                expected = True,
                test_comment=': Test manual predict (original label) accuracy: ' + str(accuracy_manual_check_original)
                             + ' on dataset with ' + str(n_rows) + ' rows data'
            ))
        return res_obj

    def test_math_function(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)
        #
        # Prepare random data
        #
        n_rows = 1000
        input_dim = 5
        # Random vectors numpy ndarray type
        data = np.random.random((n_rows, input_dim))
        #
        # Design some pattern
        # Labels are sum of the rows, then floored to the integer
        # Sum >= 0, 1, 2, 3,...
        #
        row_sums = np.sum(data, axis=1)
        labels = np.array(np.round(row_sums - 0.5, 0), dtype=int)
        max_label_value = np.max(labels)

        td = tdm.TrainingDataModel(
            x = data,
            y = labels,
            is_map_points_to_hypersphere = False
        )

        #
        # Layers Design
        #
        model_params = NetworkDesign(
            model_type      = NetworkDesign.MODEL_GENERAL_DATA,
            max_label_value = max_label_value,
            input_shape     = (td.get_x().shape[1],),
        )
        model_params.create_network()

        model_obj = NnDenseModel(
            model_params      = model_params,
            identifier_string = 'nwae_ml_unit_test_math_function',
            dir_path_model    = self.ut_params.dirpath_model,
            # Anything less than this will reduce the accuracy to < 90%
            train_epochs      = NnDenseModelUnitTest.TRAIN_EPOCHS,
            train_batch_size  = 32
        )
        model_obj.set_training_data(td=td)

        model_obj.train()

        # Load back the model
        model_obj.load_model_parameters()

        labels_categorical = to_categorical(labels)
        loss, accuracy = model_obj.evaluate(
            data   = data,
            labels = labels_categorical
        )
        accuracy_pct = round(accuracy*100, 2)
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Train Accuracy (math function): ' + str(accuracy_pct) + '% on dataset with ' + str(n_rows) + ' rows data'
        )
        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            # Lowest observed about 89.47%
            observed = (accuracy_pct > NnDenseModelUnitTest.THRESHOLD_PASS_PCT),
            expected = True,
            test_comment = ': Test train math accuracy: ' + str(accuracy_pct)
                           + ' on dataset with ' + str(n_rows) + ' rows data'
        ))

        res = self.__check_result_of_test(
            data      = data,
            labels    = labels,
            labels_original = None,
            model_obj = model_obj,
            res_obj   = res_final,
            exp_acc   = accuracy_pct
        )
        res_final.update(other_res_obj=res)
        return res_final

    def test_text_classification(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        # Training data or Documents
        # Since our labels have a max value of 88, this means in the final layer of the NN,
        # we need 89 outputs, as the NN assumes a 0 index label ends at 88, thus 89 values.
        docs_label = [
            ('잘 했어!', 11), ('잘 했어요!', 11), ('잘 한다!', 11),
            ('Молодец!', 11), ('Супер!', 11), ('Хорошо!', 11),
            ('Плохо!', 50), ('Дурак!', 50),
            ('나쁜!', 50), ('바보!', 50), ('백치!', 50), ('얼간이!', 50),
            ('미친놈', 50), ('씨발', 50), ('개', 50), ('개자식', 50),
            ('젠장', 50),
            ('ok', 88), ('fine', 88)
        ]

        res = TxtTransform(
            docs   = [x[0] for x in docs_label],
            labels = [x[1] for x in docs_label]
        ).create_padded_docs()

        x = res.padded_encoded_docs
        y = np.array(res.encoded_labels)
        y_original = np.array(res.y_original)
        x_one_hot_dict = res.x_one_hot_dict
        y_one_hot_dict = res.y_one_hot_dict
        n_rows = len(res.padded_encoded_docs)
        max_sentence_len = res.max_x_length
        max_label_value = max(res.encoded_labels)
        vocabulary_size = res.vocabulary_dimension

        td = tdm.TrainingDataModel(
            x = x,
            y = y,
            x_one_hot_dict = x_one_hot_dict,
            y_one_hot_dict = y_one_hot_dict,
            is_map_points_to_hypersphere = False
        )

        model_obj = TextModelHelper.get_model(
            model_name             = TextModelHelper.MODEL_NAME_NN_DENSE,
            identifier_string      = 'nwae_ml_unit_test_text_classification',
            dir_path_model         = self.ut_params.dirpath_model,
            training_data          = None,
            embed_max_label_val    = max_label_value,
            embed_max_sentence_len = max_sentence_len,
            embed_vocabulary_size  = vocabulary_size,
            train_epochs           = NnDenseModelUnitTest.TRAIN_EPOCHS,
            train_loss             = 'sparse_categorical_crossentropy'
        )
        model_obj.set_training_data(td=td)

        model_obj.train()

        # Load back the model
        model_obj.load_model_parameters()

        loss, accuracy = model_obj.evaluate(
            data   = x,
            labels = y
        )
        accuracy_pct = round(accuracy*100, 2)
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Train Accuracy (text classification): ' + str(accuracy_pct) + '% on dataset with '
            + str(n_rows) + ' rows data'
        )
        res_final.update_bool(res_bool=ut.UnitTest.assert_true(
            # Lowest is around 89.47%
            observed = (accuracy_pct > NnDenseModelUnitTest.THRESHOLD_PASS_PCT),
            expected = True,
            test_comment = ': Test train accuracy: ' + str(accuracy_pct)
                           + ' on dataset with ' + str(n_rows) + ' rows data'
        ))

        res = self.__check_result_of_test(
            data      = x,
            labels    = y,
            labels_original = y_original,
            model_obj = model_obj,
            res_obj   = res_final,
            exp_acc   = accuracy_pct
        )
        res_final.update(other_res_obj=res)
        return res_final

    def test_text_classification_from_sample_data(self):
        from nwae.ml.text.preprocessing.TrDataSampleData import SampleTextClassificationData
        from nwae.ml.text.preprocessing.TrDataPreprocessor import TrDataPreprocessor
        from nwae.lang.nlp.daehua.DaehuaTrainDataModel import DaehuaTrainDataModel
        import pandas as pd

        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        for sample_training_data in SampleTextClassificationData.SAMPLE_TRAINING_DATA:
            lang_main = SampleTextClassificationData.get_lang_main(sample_training_data=sample_training_data)
            lang_additional = SampleTextClassificationData.get_lang_additional(sample_training_data=sample_training_data)

            sample_data = SampleTextClassificationData.get_text_classification_training_data(
                sample_training_data = sample_training_data,
                type_io = SampleTextClassificationData.TYPE_IO_IN
            )
            expected_output_data = SampleTextClassificationData.get_text_classification_training_data(
                sample_training_data = sample_training_data,
                type_io = SampleTextClassificationData.TYPE_IO_OUT
            )

            fake_training_data = pd.DataFrame({
                DaehuaTrainDataModel.COL_TDATA_INTENT_ID: sample_data[SampleTextClassificationData.COL_CLASS],
                DaehuaTrainDataModel.COL_TDATA_INTENT_NAME: sample_data[SampleTextClassificationData.COL_CLASS_NAME],
                DaehuaTrainDataModel.COL_TDATA_TEXT: sample_data[SampleTextClassificationData.COL_TEXT],
                DaehuaTrainDataModel.COL_TDATA_TRAINING_DATA_ID: sample_data[SampleTextClassificationData.COL_TEXT_ID],
                # Don't do any processing until later
                DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED: None,
                DaehuaTrainDataModel.COL_TDATA_TEXT_LANG: None
            })
            Log.debug('Fake Training Data:\n\r' + str(fake_training_data.values))
            Log.debug('Expected Output:\n\r' + str(expected_output_data))

            trdata_pp = TrDataPreprocessor(
                model_identifier     = str([lang_main] + lang_additional) + ' Test Training Data Text Processor',
                language_main        = lang_main,
                df_training_data     = fake_training_data,
                dirpath_wordlist     = self.ut_params.dirpath_wordlist,
                postfix_wordlist     = self.ut_params.postfix_wordlist,
                dirpath_app_wordlist = self.ut_params.dirpath_app_wordlist,
                postfix_app_wordlist = self.ut_params.postfix_app_wordlist,
                dirpath_synonymlist  = self.ut_params.dirpath_synonymlist,
                postfix_synonymlist  = self.ut_params.postfix_synonymlist,
                reprocess_all_text   = True,
                languages_additional = lang_additional
            )

            df_pp_data, embed_params = trdata_pp.preprocess_training_data_text()

            Log.debug('*********** FINAL SEGMENTED DATA (' + str(trdata_pp.df_training_data.shape[0]) + ' sentences)')
            Log.debug(df_pp_data.columns)
            Log.debug(df_pp_data.values)

            x = embed_params.x
            y = embed_params.y
            x_one_hot_dict = embed_params.x_one_hot_dict
            y_one_hot_dict = embed_params.y_one_hot_dict
            n_rows = len(x)
            max_sentence_len = embed_params.max_sent_len
            max_label_value = embed_params.max_label_val
            vocabulary_size = embed_params.vocab_size

            td_model_obj = TextTrainer.convert_preprocessed_text_to_training_data_model(
                model_name         = NnDenseModel.MODEL_NAME,
                training_dataframe = df_pp_data,
                embedding_x        = x,
                embedding_y        = y,
                embedding_x_one_hot_dict = x_one_hot_dict,
                embedding_y_one_hot_dict = y_one_hot_dict,
            )
            model_obj = TextModelHelper.get_model(
                model_name             = TextModelHelper.MODEL_NAME_NN_DENSE,
                identifier_string      = 'nwae_ml_unit_test_text_classification',
                dir_path_model         = self.ut_params.dirpath_model,
                training_data          = None,
                embed_max_label_val    = max_label_value,
                embed_max_sentence_len = max_sentence_len,
                embed_vocabulary_size  = vocabulary_size,
                train_epochs           = NnDenseModelUnitTest.TRAIN_EPOCHS,
                train_loss             = 'sparse_categorical_crossentropy'
            )
            model_obj.set_training_data(td=td_model_obj)

            model_obj.train()

            # Load back the model
            model_obj.load_model_parameters()

            loss, accuracy = model_obj.evaluate(
                data   = x,
                labels = y
            )
            accuracy_pct = round(accuracy * 100, 2)
            Log.info(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Train Accuracy (lang ' + str(lang_main) + '): ' + str(accuracy_pct) + '% on dataset with '
                + str(n_rows) + ' rows data'
            )
            res_final.update_bool(res_bool=ut.UnitTest.assert_true(
                # Lowest is around 89.47%
                observed     = (accuracy_pct > NnDenseModelUnitTest.THRESHOLD_PASS_PCT_LOW),
                expected     = True,
                test_comment = ': Lang ' + str(lang_main) + ', test train accuracy: ' + str(accuracy_pct)
                             + ' on dataset with ' + str(n_rows) + ' rows data'
            ))

        return res_final


if __name__ == '__main__':
    import nwae.ml.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_DEFAULT
    )
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO

    ut_params = ut.UnitTestParams(
        dirpath_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_DIR_WORDLIST),
        postfix_wordlist     = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_WORDLIST),
        dirpath_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_APP_WORDLIST),
        postfix_app_wordlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
        dirpath_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
        postfix_synonymlist  = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
        dirpath_model        = config.get_config(param=cf.Config.PARAM_MODEL_DIR)
    )
    Log.important('Unit Test Params: ' + str(ut_params.to_string()))

    res_ut = NnDenseModelUnitTest(ut_params=ut_params).run_unit_test()
    exit(res_ut.count_fail)
