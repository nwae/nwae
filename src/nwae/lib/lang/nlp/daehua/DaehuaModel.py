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
    # e.g. 10:12:36, 12:15
    DAEHUA_MODEL_TYPE_TIME   = 'time'

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
            # Replace '-lt' with '<'
            formula_str_encoding = re.sub(pattern='-lt', repl='<', string=formula_str_encoding)
            # Replace '-gt' with '>'
            formula_str_encoding = re.sub(pattern='-gt', repl='>', string=formula_str_encoding)

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

            #
            # Default to search the front value first
            # TODO Make this more intelligent
            #
            value = DaehuaModel.get_var_value_front(
                var_name = var,
                string = s,
                var_type_names = names,
                data_type = data_type
            )
            if not value:
                value = DaehuaModel.get_var_value_back(
                    var_name = var,
                    string = s,
                    var_type_names = names,
                    data_type = data_type
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
                    elif data_type == DaehuaModel.DAEHUA_MODEL_TYPE_TIME:
                        var_values[var] = value
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
        if patterns_list is None:
            lg.Log.error(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': No patterns list provided for string "' + str(string)
                + '", var name "' + str(var_name) + '".'
            )
            return None

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
            var_type_names,
            data_type
    ):
        var_type_names = var_type_names.lower()
        # Always check float first
        pattern_check_front_float = '.*[^0-9\-]+([+\-]*[0-9]+[.][0-9]*)[ ]*(' + var_type_names + ').*'
        pattern_check_front_float_start = '^([+\-]*[0-9]+[.][0-9]*)[ ]*(' + var_type_names + ').*'
        pattern_check_front_int = '.*[^0-9\-]+([+\-]*[0-9]+)[ ]*(' + var_type_names + ').*'
        pattern_check_front_int_start = '^([+\-]*[0-9]+)[ ]*(' + var_type_names + ').*'
        # Time pattern. e.g. 12:30:59, 23:45
        # Check HHMMSS first, if that fails then only HHMM
        pattern_check_front_time_HHMMSS = '.*[^0-9]+([0-9]+[:][0-9]+[:][0-9]+)[ ]*(' + var_type_names + ').*'
        pattern_check_front_time_start_HHMMSS = '^([0-9]+[:][0-9]+[:][0-9]+)[ ]*(' + var_type_names + ').*'
        pattern_check_front_time_HHMM = '.*[^0-9]+([0-9]+[:][0-9]+)[ ]*(' + var_type_names + ').*'
        pattern_check_front_time_start_HHMM = '^([0-9]+[:][0-9]+)[ ]*(' + var_type_names + ').*'

        patterns_list = None
        if data_type in (DaehuaModel.DAEHUA_MODEL_TYPE_FLOAT, DaehuaModel.DAEHUA_MODEL_TYPE_INT):
            patterns_list = (
                    pattern_check_front_float, pattern_check_front_float_start,
                    pattern_check_front_int, pattern_check_front_int_start
            )
        elif data_type == DaehuaModel.DAEHUA_MODEL_TYPE_TIME:
            patterns_list = (
                pattern_check_front_time_HHMMSS, pattern_check_front_time_start_HHMMSS,
                pattern_check_front_time_HHMM, pattern_check_front_time_start_HHMM
            )

        m = DaehuaModel.get_var_value_regex(
            # Always check float first
            patterns_list = patterns_list,
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
            var_type_names,
            data_type
    ):
        var_type_names = var_type_names.lower()
        # Always check float first
        pattern_check_back_float = '.*(' + var_type_names + ')[ ]*([+\-]*[0-9]+[.][0-9]*).*'
        pattern_check_back_int = '.*(' + var_type_names + ')[ ]*([+\-]*[0-9]+).*'
        # Time pattern. e.g. 12:30:59, 23:45
        # Check HHMMSS first, if that fails then only HHMM
        pattern_check_back_time_HHMMSS = '.*(' + var_type_names + ')[ ]*([0-9]+[:][0-9]+[:][0-9]+).*'
        pattern_check_back_time_HHMM = '.*(' + var_type_names + ')[ ]*([0-9]+[:][0-9]+).*'

        patterns_list = None
        if data_type in (DaehuaModel.DAEHUA_MODEL_TYPE_FLOAT, DaehuaModel.DAEHUA_MODEL_TYPE_INT):
            patterns_list = (pattern_check_back_float, pattern_check_back_int)
        elif data_type == DaehuaModel.DAEHUA_MODEL_TYPE_TIME:
            patterns_list = (pattern_check_back_time_HHMMSS, pattern_check_back_time_HHMM)

        m = DaehuaModel.get_var_value_regex(
            # Always check float first
            patterns_list = patterns_list,
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

        if calc_result is not None:
            calc_result = round(calc_result, DaehuaModel.DEFAULT_NUMBER_ROUNDING)

        class answer_result:
            def __init__(self, answer_value, variable_values):
                self.answer_value = answer_value
                self.variable_values = variable_values

        return answer_result(
            answer_value    = calc_result,
            variable_values = var_values
        )


if __name__ == '__main__':
    # cf_obj = cf.Config.get_cmdline_params_and_init_config_singleton(
    #     Derived_Class = cf.Config
    # )
    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_INFO

    tests = [
        {
            'encoding': 'Volume of Sphere -*-vars==r,float,radius&r;d,float,diameter&d' \
                        + '::' + 'answer==(4/3)*(3.141592653589793 * $$r*$$r*$$r)-*-',
            'questions': [
                'What is the volume of a sphere of radius 5.88?'
            ]
        },
        {
            'encoding': '-*-'\
                   + 'vars==id,float,id&indo' \
                   + '::'\
                   + 'answer==('\
                   + '  ($$id -lt 0)*1*(1 + (1 | (-$$id)))'\
                   + '+ ($$id -gt 0)*1*(1 + $$id)'\
                   + ')-*-',
            'questions': [
                'What is -2.6 indo odds?',
                'What is +1.2 indo odds?'
            ]
        },
        {
            'encoding': '-*-vars==acc,int,尾号&账号;m,int,月;d,int,日;t,time,完成;amt,float,民币&币;bal,float,余额::answer==$$amt-*-',
            'questions': [
                '【中国农业银行】您尾号0579账户10月17日09:27完成代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】您尾号0579账户10月17日09:27:55完成代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】您尾号0579账户10月17日完成09:27代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】您尾号0579账户10月17日完成09:27:55代付交易人民币2309.95，余额2932.80。'
            ]
        }
    ]

    for test in tests:
        encoding = test['encoding']
        questions = test['questions']

        for question in questions:
            cmobj = DaehuaModel(
                encoding_str = encoding,
                question     = question
            )
            result = cmobj.get_answer()
            print(
                'Answer = ' + str(result.answer_value)
                + ', variables = ' + str(result.variable_values)
            )
