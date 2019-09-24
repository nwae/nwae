# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import re
import pandas as pd
import nwae.utils.StringUtils as su


#
# The first thing we need in a conversation model is a "daehua training data language"
# to encode various information required for reply processing parameters extracted
# from a question.
#
# Daehua Training Language Description
#   TODO Move the variable description away from the Intent Name column to a separate column
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

    def remove_variables_and_variable_declarations(
            self,
            str_list
    ):
        clean_1_list = self.remove_variables(
            var_list = str_list
        )
        clean_2_list = self.remove_variable_declaration(
            decl_list = clean_1_list
        )
        return clean_2_list

    #
    # In the training data, when we want to specify it is a variable, we put "$$" in front
    # of a variable name. This will remove all variables specified.
    #
    def remove_variables(
            self,
            var_list
    ):
        try:
            # Remove anything of the form $$var, $$xyz,...
            clean_list = [re.sub(pattern='[$]{2}[a-z0-9]+', repl='', string=str(s)) for s in var_list]
            clean_list = [re.sub(pattern='[ ]+', repl=' ', string=s) for s in clean_list]
            clean_list = [su.StringUtils.trim(s) for s in clean_list]
            return clean_list
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception "' + str(ex) + '" cleaning var list ' + str(var_list)
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    #
    # Now we encode the variable names, types, etc. together with the intent name.
    # TODO Move them to a separate column
    #
    def remove_variable_declaration(
            self,
            # Can be string or list type
            decl_list
    ):
        try:
            is_type_str = type(decl_list) is str
            if is_type_str:
                decl_list = [decl_list]

            clean_decl_list = [re.sub(pattern='[ ]*[-*-].*[-*-]', repl='', string=str(s)) for s in decl_list]
            clean_decl_list = [re.sub(pattern='[ ]+', repl=' ', string=s) for s in clean_decl_list]
            clean_decl_list = [su.StringUtils.trim(s) for s in clean_decl_list]

            if is_type_str:
                return clean_decl_list[0]
            else:
                return clean_decl_list
        except Exception as ex:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Exception "' + str(ex) + '" cleaning var declaration list ' + str(decl_list)
            lg.Log.error(errmsg)
            raise Exception(errmsg)

    #
    # Here we clean the daehua training data language, and leave only pure training data.
    # It can also be just plain training data of which we let it pass.
    #
    def clean_daehua_training_data(
            self
    ):
        df_daehua_processed = pd.DataFrame(self.daehua_training_data)
        #
        # Process by intent
        #
        # lg.Log.debugdebug(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Training Data:\n\r' + str(df_daehua_processed.values)
        # )

        #
        # Clean intent names
        #
        intent_names_list = df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME].tolist()
        df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_INTENT_NAME] =\
            self.remove_variable_declaration(decl_list = intent_names_list)

        #
        # Clean variables from txt and txt segmented
        #
        text_list = df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_TEXT].tolist()
        df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_TEXT] =\
            self.remove_variables_and_variable_declarations(str_list = text_list)

        text_seg_list = df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED].tolist()
        df_daehua_processed[DaehuaTrainDataModel.COL_TDATA_TEXT_SEGMENTED] =\
            self.remove_variables_and_variable_declarations(str_list = text_seg_list)

        # lg.Log.debugdebug(
        #     str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #     + ': Training Data:\n\r' + str(df_daehua_processed.values)
        # )

        return df_daehua_processed


if __name__ == '__main__':
    exit(0)
