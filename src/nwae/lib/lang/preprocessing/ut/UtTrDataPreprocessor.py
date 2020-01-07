# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.lib.lang.LangFeatures as lf
from nwae.lib.lang.nlp.daehua.DaehuaTrainDataModel import DaehuaTrainDataModel
from nwae.lib.lang.preprocessing.TrDataPreprocessor import TrDataPreprocessor
from nwae.utils.UnitTest import ResultObj
import pandas as pd
from nwae.samples.SampleTextClassificationData import SampleTextClassificationData


class UtTrDataPreprocessor:

    def __init__(self):
        return

    def run_unit_test(
            self,
            lang
    ):
        sample_data = SampleTextClassificationData.get_text_classification_training_data(
            lang=lang
        )

        fake_training_data = pd.DataFrame({
            DaehuaTrainDataModel.COL_TDATA_INTENT_ID: sample_data[SampleTextClassificationData.COL_CLASS],
            DaehuaTrainDataModel.COL_TDATA_INTENT_NAME: sample_data[SampleTextClassificationData.COL_CLASS_NAME],
            DaehuaTrainDataModel.COL_TDATA_TEXT: sample_data[SampleTextClassificationData.COL_TEXT],
            DaehuaTrainDataModel.COL_TDATA_TRAINING_DATA_ID: sample_data[SampleTextClassificationData.COL_TEXT_ID],
            # Don't do any processing until later
            DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED: None
        })
        Log.debug('Fake Training Data:\n\r' + str(fake_training_data))

        ctdata = TrDataPreprocessor(
            model_identifier = str(lang) + ' Test Training Data Text Processor',
            language         = lang,
            df_training_data = fake_training_data,
            dirpath_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_WORDLIST),
            postfix_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_WORDLIST),
            dirpath_app_wordlist = config.get_config(param=Config.PARAM_NLP_DIR_APP_WORDLIST),
            postfix_app_wordlist = config.get_config(param=Config.PARAM_NLP_POSTFIX_APP_WORDLIST),
            dirpath_synonymlist  = config.get_config(param=Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synonymlist  = config.get_config(param=Config.PARAM_NLP_POSTFIX_SYNONYMLIST),
            reprocess_all_text   = True,
        )

        ctdata.go()

        Log.debug('*********** FINAL SEGMENTED DATA (' + str(ctdata.df_training_data.shape[0]) + ' sentences)')
        Log.debug(ctdata.df_training_data.columns)
        Log.debug(ctdata.df_training_data.values)

        Log.debug('*********** ROWS CHANGED ***********')
        count = 0
        for row in ctdata.list_of_rows_with_changed_processed_text:
            count += 1
            print(str(count) + '. ' + str(row))


if __name__ == '__main__':
    from nwae.config.Config import Config
    config = Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class=Config,
        default_config_file='/usr/local/git/nwae/nwae/app.data/config/local.nwae.cf'
    )

    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    UtTrDataPreprocessor().run_unit_test(
        lang = lf.LangFeatures.LANG_KO
    )
    UtTrDataPreprocessor().run_unit_test(
        lang = lf.LangFeatures.LANG_VN
    )
