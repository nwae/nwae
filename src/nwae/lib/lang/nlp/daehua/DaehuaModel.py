# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.config.Config as cf
import re


class DaehuaModel:

    DAEHUA_MODEL_ENCODING_TYPE = 'type'
    DAEHUA_MODEL_ENCODING_NAMES = 'names'

    DAEHUA_MODEL_TYPE_STRING = 'str'
    DAEHUA_MODEL_TYPE_FLOAT  = 'float'
    DAEHUA_MODEL_TYPE_INT    = 'int'

    @staticmethod
    def get_convmodel_encoding_str(
            s
    ):
        try:
            m = re.match(pattern='.*[-][*][-](.*)[-][*][-].*', string=s)
            str_encoding = m.group(1)
            return str_encoding
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get encoding string for "' + str(s) + '". Exception ' + str(ex) + '.'
            lg.Log.error(errmsg)
            return None

    @staticmethod
    def get_var_encoding(
            s
    ):
        try:
            var_encoding = {}

            str_encoding = DaehuaModel.get_convmodel_encoding_str(
                s = s
            )
            str_encoding = str_encoding.split(';')
            for varset in str_encoding:
                var_desc = varset.split(',')
                var_encoding[var_desc[0]] = {
                    DaehuaModel.DAEHUA_MODEL_ENCODING_TYPE: var_desc[1],
                    DaehuaModel.DAEHUA_MODEL_ENCODING_NAMES: var_desc[2].split('&')
                }
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
            s,
            var_encoding,
            var_values
    ):
        try:
            code = ''
            formula_str_encoding = DaehuaModel.get_convmodel_encoding_str(
                s = s
            )

            d = var_values
            for var in var_encoding:
                formula_str_encoding = re.sub(
                    pattern = '[$]' + str(var),
                    repl    = 'd[\'' + str(var) + '\']',
                    string  = formula_str_encoding
                )

            lg.Log.info(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Code for s "' + str(s) + '" var encoding ' + str(var_encoding)
                + ', values ' + str(var_values)
                + '\n\r   ' + code
            )

            return formula_str_encoding
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get formula encoding string for "' + str(s)\
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
        lg.Log.info(
            str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Extracting vars from "' + str(s) + '", using encoding ' + str(var_encoding)
        )

        var_values = {}

        # Look one by one
        for var in var_encoding.keys():
            var_values[var] = None
            # Get the names and join them using '|' for matching regex
            names = '|'.join(var_encoding[var][DaehuaModel.DAEHUA_MODEL_ENCODING_NAMES])
            pattern = '.*([0-9]*)[ ]*(' + names + ')[ ]*([0-9]*).*'
            lg.Log.info(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': For var "' + str(var) + '" using match pattern "' + str(pattern) + '"..'
            )
            m = re.match(pattern=pattern, string=s)
            if m:
                groups = m.groups()
                lg.Log.info(
                    str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': For var "' + str(var) + '" found groups ' + str(groups)
                )
                try:
                    if groups[0] != '':
                        var_values[var] = int(groups[0])
                    else:
                        var_values[var] = int(groups[2])
                except Exception as ex_int_conv:
                    errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                             + ': Failed to extract variable "' + str(var) + '" from "' + str(s)\
                             + '". Exception ' + str(ex_int_conv) + '.'
                    lg.Log.warning(errmsg)

        lg.Log.info(
            str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': For s "' + str(s) + '" var values ' + str(var_values)
        )

        return var_values

    def __init__(
            self,
            intent_name,
            question,
            answer
    ):
        self.intent_name = intent_name
        self.question = question
        self.answer = answer
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + '\n\r   Intent Name "' + str(intent_name)
            + '"\n\r   question "' + str(question)
            + '"\n\r   answer "' + str(answer) + '"'
        )
        return

    def get_answer(self):
        var_encoding = DaehuaModel.get_var_encoding(
            s = self.intent_name
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Var Encoding:\n\r' + str(var_encoding)
        )

        #
        # Extract variables from question
        #
        var_values = DaehuaModel.extract_variable_values(
            s = self.question,
            var_encoding = var_encoding
        )

        #
        # Extract formula from answer
        #
        formula_code_str = DaehuaModel.get_formula_code_str(
            s = self.answer,
            var_encoding = var_encoding,
            var_values = var_values
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Evaluating code: ' + str(formula_code_str)
        )
        d = var_values
        result = eval(formula_code_str)
        lg.Log.info(
            'Result = ' + str(result)
        )
        return result


if __name__ == '__main__':
    # cf_obj = cf.Config.get_cmdline_params_and_init_config_singleton(
    #     Derived_Class = cf.Config
    # )
    lg.Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_1

    intent_name = 'Volume of Sphere -*-r,float,radius&r;d,float,diameter&d-*-'
    question = 'What is the volume of a sphere of radius 5?'
    answer = 'Your answer is -*-(4/3)*(3.141592653589793 * $r*$r*$r)-*-'

    cmobj = DaehuaModel(
        intent_name = intent_name,
        question    = question,
        answer      = answer
    )
    result = cmobj.get_answer()
    print('Answer = ' + str(result))
