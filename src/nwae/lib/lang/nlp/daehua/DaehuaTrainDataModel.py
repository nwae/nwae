# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.config.Config as cf
import nwae.lib.lang.LangFeatures  as lf
import json


#
# The first thing we need in a conversation model is a "training data language"
# to encode various information required for reply processing parameters extracted
# from a question.
#
class DaehuaTrainDataModel:

    #
    # Expected columns in a data frame to be passed in to this class
    #
    COL_TDATA_TRAINING_DATA_ID = 'Training Data Id'
    COL_TDATA_CATEGORY         = 'Category'
    COL_TDATA_INTENT_NAME      = 'Intent Name'
    COL_TDATA_INTENT_ID        = 'Intent ID'
    COL_TDATA_INTENT_TYPE      = 'Intent Type'
    COL_TDATA_INTENT_INDEX     = 'Intent Index'    # Derived Column for indexing training data sentences
    COL_TDATA_TEXT             = 'Text'
    COL_TDATA_TEXT_LENGTH      = 'Text Length'     # Derived Column
    COL_TDATA_TEXT_SEGMENTED   = 'Text Segmented'  # Derived Column

    def __init__(
            self,
            # In pandas data frame with the above columns
            training_data
    ):
        self.training_data = training_data
        return


if __name__ == '__main__':
    exit(0)