# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.config.Config as cf
import re


#
# The first thing we need in a conversation model is a "daehua language"
# to encode various information required for reply processing parameters extracted
# from a question.
#
# Daehua Language Description
#   Encoding string can be of the form:
#
#      'Calculate Energy -*-vars==m,float,mass&m;c,float,light&speed::answer==$$m * ($$c * $$c)-*-
#
#   where 'm' is a variable name of type float, and 'c' (speed of light) is another
#   variable name of the same type.
#   For now we support str, float, int. We don't support specific regex to not complicate
#   things.
#
#   Then training data may be as such:
#     "Help me calculate energy, my mass is $$m, and light speed $$c."
#     "Calculate energy for me, mass $$m, c $$c."
#
#   And the answer may be encoded similarly using "-*-" as delimiter:
#     Your answer is -*-$$m * ($$c * $$c)-*-
#
#

class DaehuaModel:
    DEFAULT_NUMBER_ROUNDING = 5

    DAEHUA_MODEL_ENCODE_STR    = 'encode_str'
    DAEHUA_MODEL_OBJECT_VARS   = 'vars'
    DAEHUA_MODEL_OBJECT_ANSWER = 'answer'

    DAEHUA_MODEL_OBJECT_VARS_TYPE = 'type'
    DAEHUA_MODEL_OBJECT_VARS_NAMES = 'names'

    # We use '-*-' to open and close the encoding language
    DAEHUA_MODEL_ENCODING_CHARS_START_END = '[-][*][-](.*)[-][*][-]'

    # Separates the vars and answer object. E.g.
    #    vars==m,float,mass&m;c,float,light&speed::answer==$$m * ($$c * $$c)
    DAEHUA_MODEL_OBJECT_SEPARATOR = '::'
    DAEHUA_MODEL_OBJECT_DEFINITION_SYMBOL = '=='
    # Separates the different variables definition. e.g. 'm,float,mass&m;c,float,light&speed'
    DAEHUA_MODEL_VAR_DEFINITION_SEPARATOR = ';'
    # Separates the description of the same variable. e.g. 'm,float,mass&m'
    DAEHUA_MODEL_VAR_DESCRIPTION_SEPARATOR = ','
    # Separates the names of a variable. e.g. 'mass&m'
    DAEHUA_MODEL_VAR_NAMES_SEPARATOR = '&'
    DAEHUA_MODEL_VAR_MARKUP_IN_QUESTION = '[$]{2}'

    DAEHUA_MODEL_TYPE_FLOAT  = 'float'
    DAEHUA_MODEL_TYPE_INT    = 'int'

    #
    # Returns the string encoding of the model
    #
    @staticmethod
    def get_daehua_model_encoding_str(
            s
    ):
        try:
            daehua_model_encoding_str = {
                DaehuaModel.DAEHUA_MODEL_ENCODE_STR: None,
                DaehuaModel.DAEHUA_MODEL_OBJECT_VARS: None,
                DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER: None
            }

            m = re.match(pattern='.*'+DaehuaModel.DAEHUA_MODEL_ENCODING_CHARS_START_END+'.*', string=s)
            dh_encode_str = m.group(1)
            daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_ENCODE_STR] = dh_encode_str

            # Split by '::'
            dh_objects_str = dh_encode_str.split(DaehuaModel.DAEHUA_MODEL_OBJECT_SEPARATOR)
            for dh_obj_str in dh_objects_str:
                # Break again
                parts = dh_obj_str.split(sep=DaehuaModel.DAEHUA_MODEL_OBJECT_DEFINITION_SYMBOL)
                if parts[0] == DaehuaModel.DAEHUA_MODEL_OBJECT_VARS:
                    daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_OBJECT_VARS] = parts[1]
                elif parts[0] == DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER:
                    daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER] = parts[1]

            lg.Log.info(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Decoded encoding parts string: ' + str(daehua_model_encoding_str)
            )
            return daehua_model_encoding_str
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get encoding string for "' + str(s) + '". Exception ' + str(ex) + '.'
            lg.Log.error(errmsg)
            return None

    #
    # Extract from string encoding 'm,float,mass&m;c,float,light&speed' into something like:
    #   {
    #      'm': {
    #         'type': 'float',
    #         'names': ['mass', 'm']
    #      },
    #      'c': {
    #         'type': 'float',
    #         'names': ['speed', 'light']
    #      }
    #   }
    #
    @staticmethod
    def decode_vars_object_str(
            s
    ):
        try:
            var_encoding = {}

            # Here we split "m,float,mass&m;c,float,light&speed" into ['m,float,mass&m', 'c,float,light&speed']
            str_encoding = s.split(DaehuaModel.DAEHUA_MODEL_VAR_DEFINITION_SEPARATOR)
            for varset in str_encoding:
                # Here we split 'm,float,mass&m' into ['m','float','mass&m']
                var_desc = varset.split(DaehuaModel.DAEHUA_MODEL_VAR_DESCRIPTION_SEPARATOR)

                part_var_id = var_desc[0]
                part_var_type = var_desc[1]
                part_var_names = var_desc[2]

                var_encoding[part_var_id] = {
                    # Extract 'float' from ['m','float','mass&m']
                    DaehuaModel.DAEHUA_MODEL_OBJECT_VARS_TYPE: part_var_type,
                    # Extract ['mass','m'] from 'mass&m'
                    DaehuaModel.DAEHUA_MODEL_OBJECT_VARS_NAMES: part_var_names.split(
                        sep = DaehuaModel.DAEHUA_MODEL_VAR_NAMES_SEPARATOR
                    )
                }
                lg.Log.info(
                    str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Successfully decoded vars object item "'
                    + str(part_var_id) + '": ' + str(var_encoding[var_desc[0]])
                )
            return var_encoding
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get var encoding for "' + str(s) + '". Exception ' + str(ex) + '.'
            lg.Log.error(errmsg)
            return None

    #
    # Returns the code ready for python eval() call
    #
    @staticmethod
    def get_formula_code_str(
            daehua_answer_object_str,
            var_values
    ):
        try:
            formula_str_encoding = daehua_answer_object_str
            # TODO This is a hardcode, remove in the future
            # Replace '|' with divide '/'
            formula_str_encoding = re.sub(pattern='[|]', repl='/', string=formula_str_encoding)

            d = var_values
            for var in var_values:
                formula_str_encoding = re.sub(
                    # Replace variables in question such as $$mass with a dictionary value d['mass']
                    pattern = DaehuaModel.DAEHUA_MODEL_VAR_MARKUP_IN_QUESTION + str(var),
                    # 'd' is our default dictionary object, so to make the eval() run, we must
                    # first define d = var_values
                    repl    = 'd[\'' + str(var) + '\']',
                    string  = formula_str_encoding
                )

            lg.Log.debug(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Code for answer object string "' + str(daehua_answer_object_str)
                + '", values ' + str(var_values)
                + '\n\r   ' + str(formula_str_encoding)
            )

            return formula_str_encoding
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get formula encoding string for "' + str(daehua_answer_object_str)\
                     + '". Exception ' + str(ex) + '.'
            lg.Log.error(errmsg)
            return None

    #
    # Extract variables from string
    #
    @staticmethod
    def extract_variable_values(
            s,
            var_encoding
    ):
        s = str(s).lower()

        lg.Log.debug(
            str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Extracting vars from "' + str(s) + '", using encoding ' + str(var_encoding)
        )

        var_values = {}

        # Look one by one
        for var in var_encoding.keys():
            var_values[var] = None
            # Get the names and join them using '|' for matching regex
            names = '|'.join(var_encoding[var][DaehuaModel.DAEHUA_MODEL_OBJECT_VARS_NAMES])
            data_type = var_encoding[var][DaehuaModel.DAEHUA_MODEL_OBJECT_VARS_TYPE]

            value = DaehuaModel.get_var_value_front(
                var_name = var,
                string = s,
                var_type_names = names
            )
            if not value:
                value = DaehuaModel.get_var_value_back(
                    var_name = var,
                    string = s,
                    var_type_names = names
                )

            if value:
                lg.Log.debug(
                    str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': For var "' + str(var) + '" found value ' + str(value)
                )
                try:
                    if data_type == DaehuaModel.DAEHUA_MODEL_TYPE_INT:
                        var_values[var] = int(value)
                    elif data_type == DaehuaModel.DAEHUA_MODEL_TYPE_FLOAT:
                        var_values[var] = float(value)
                    else:
                        raise Exception('Unrecognized type "' + str(data_type) + '".')
                except Exception as ex_int_conv:
                    errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Failed to extract variable "' + str(var) + '" from "' + str(s)\
                             + '". Exception ' + str(ex_int_conv) + '.'
                    lg.Log.warning(errmsg)

        lg.Log.debug(
            str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': For s "' + str(s) + '" var values ' + str(var_values)
        )

        return var_values

    @staticmethod
    def get_var_value_regex(
            patterns_list,
            var_name,
            string
    ):
        lg.Log.debug(
            str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': For var "' + str(var_name)
            + '" using match patterns list ' + str(patterns_list)
        )
        for pattern in patterns_list:
            m = re.match(pattern=pattern, string=string)
            if m:
                lg.Log.debug(
                    str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': For var "' + str(var_name) + '" using pattern "' + str(pattern)
                    + '", found groups ' + str(m.groups())
                )
                return m
        return None

    @staticmethod
    def get_var_value_front(
            var_name,
            string,
            var_type_names
    ):
        # Always check float first
        pattern_check_front_float = '.*[^0-9]+([0-9]+[.][0-9]*)[ ]*(' + var_type_names + ').*'
        pattern_check_front_float_start = '$([0-9]+[.][0-9]*)[ ]*(' + var_type_names + ').*'
        pattern_check_front_int = '.*[^0-9]+([0-9]+)[ ]*(' + var_type_names + ').*'
        pattern_check_front_int_start = '$([0-9]+)[ ]*(' + var_type_names + ').*'

        m = DaehuaModel.get_var_value_regex(
            # Always check float first
            patterns_list = (
                pattern_check_front_float, pattern_check_front_float_start,
                pattern_check_front_int, pattern_check_front_int_start),
            var_name      = var_name,
            string        = string
        )
        if m:
            return m.group(1)
        return None

    @staticmethod
    def get_var_value_back(
            var_name,
            string,
            var_type_names
    ):
        # Always check float first
        pattern_check_back_float = '.*(' + var_type_names + ')[ ]*([0-9]+[.][0-9]*).*'
        pattern_check_back_int = '.*(' + var_type_names + ')[ ]*([0-9]+).*'

        m = DaehuaModel.get_var_value_regex(
            # Always check float first
            patterns_list = (pattern_check_back_float, pattern_check_back_int),
            var_name      = var_name,
            string        = string
        )
        if m:
            return m.group(2)
        return None

    def __init__(
            self,
            encoding_str,
            question
    ):
        self.encoding_str = encoding_str
        self.question = question
        lg.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Daehua Model Encoding "' + str(self.encoding_str)
            + '" question "' + str(question) + '".'
        )
        #
        # Decode the model variables, answer
        #
        self.daehua_model_str = None
        self.daehua_model_obj_vars = None
        self.__decode_str()
        return

    def __decode_str(self):
        self.daehua_model_str = DaehuaModel.get_daehua_model_encoding_str(
            s = self.encoding_str
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Model Encoding strings: ' + str(self.daehua_model_str)
        )
        self.daehua_model_obj_vars = DaehuaModel.decode_vars_object_str(
            s = self.daehua_model_str[DaehuaModel.DAEHUA_MODEL_OBJECT_VARS]
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Model Object vars: ' + str(self.daehua_model_obj_vars)
        )
        return

    def get_answer(self):
        #
        # Extract variables from question
        #
        var_values = DaehuaModel.extract_variable_values(
            s = self.question,
            var_encoding = self.daehua_model_obj_vars
        )

        #
        # Extract formula from answer
        #
        formula_code_str = DaehuaModel.get_formula_code_str(
            daehua_answer_object_str = self.daehua_model_str[DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER],
            var_values = var_values
        )
        calc_result = None
        try:
            lg.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Evaluating code: ' + str(formula_code_str)
                + ' for variables ' + str(var_values)
            )
            d = var_values
            calc_result = eval(formula_code_str)
            lg.Log.debug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Result = ' + str(calc_result)
            )
        except Exception as ex_eval:
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error evaluating formula code "' + str(formula_code_str)\
                     + '" for var values ' + str(var_values)\
                     + '. Exception ' + str(ex_eval) + '.'
            lg.Log.error(errmsg)
            calc_result = None

        class answer_result:
            def __init__(self, answer_value, variable_values):
                self.answer_value = answer_value
                self.variable_values = variable_values

        return answer_result(
            answer_value    = round(calc_result,DaehuaModel.DEFAULT_NUMBER_ROUNDING),
            variable_values = var_values
        )


if __name__ == '__main__':
    # cf_obj = cf.Config.get_cmdline_params_and_init_config_singleton(
    #     Derived_Class = cf.Config
    # )
    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_2

    encoding = 'Volume of Sphere -*-vars==r,float,radius&r;d,float,diameter&d'\
                  + '::'\
                  + 'answer==(4/3)*(3.141592653589793 * $$r*$$r*$$r)-*-'
    question = 'What is the volume of a sphere of radius 5.88?'
    encoding = '-*-vars==odds,float,odds&odd&hk::answer==$$odds+1-*-'
    question = 'What is 1.5 HK 1.9 in decimal?'

    cmobj = DaehuaModel(
        encoding_str = encoding,
        question     = question
    )
    result = cmobj.get_answer()
    print(
        'Answer = ' + str(result.answer_value)
        + ', variables = ' + str(result.variable_values)
    )
