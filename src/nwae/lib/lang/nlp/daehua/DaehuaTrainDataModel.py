# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.config.Config as cf
import nwae.lib.lang.LangFeatures  as lf
import json


#
# The first thing we need in a conversation model is a "daehua training data language"
# to encode various information required for reply processing parameters extracted
# from a question.
#
# Daehua Training Language Description
#   Intent Name can be of the form 'Calculate Energy -*-mass_float,c_float-*-
#   where 'mass' is a variable name of type float, and 'c' (speed of light) is another
#   variable name of the same type.
#   For now we support str, float, int. We don't support specific regex to not complicate
#   things.
#
#   Then training data may be as such:
#     "Help me calculate energy, my mass is $$mass, and light speed $$c."
#     "Calculate energy for me, mass $$mass, c $$c."
#
#   And the answer may be encoded as such:
#     Your answer is $$mass * ($$c * $$c)
#
#
class DaehuaTrainDataModel:

    DAEHUA_TR_LANG_MODEL_TYPE_STRING = 'str'
    DAEHUA_TR_LANG_MODEL_TYPE_FLOAT  = 'float'
    DAEHUA_TR_LANG_MODEL_TYPE_INT    = 'int'

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
            daehua_training_data
    ):
        self.daehua_training_data = daehua_training_data
        return

    #
    # Here we decode the daehua training data language.
    # It can also be just plain training data of which we let it pass.
    #
    def process_daehua_training_data(
            self
    ):
        df_daehua_processed = self.daehua_training_data
        #
        # Process by intent ID
        #

        return df_daehua_processed


if __name__ == '__main__':
    exit(0)
