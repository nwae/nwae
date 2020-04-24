# -*- coding: utf-8 -*-

import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import re
import nwae.utils.StringUtils as su
import mex.MatchExpression as mex


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

    # We use '-*-' to open and close the encoding language
    DAEHUA_MODEL_ENCODING_CHARS_START_END = '[-][*][-](.*)[-][*][-]'

    # Separates the vars and answer object. E.g.
    #    vars==m,float,mass&m;c,float,light&speed::answer==$$m * ($$c * $$c)
    DAEHUA_MODEL_OBJECT_SEPARATOR = '::'
    DAEHUA_MODEL_OBJECT_DEFINITION_SYMBOL = '=='
    DAEHUA_MODEL_VAR_MARKUP_IN_QUESTION = '[$]{2}'

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

            #
            # Decode entire Daehua pattern
            #
            m = re.match(pattern='.*'+DaehuaModel.DAEHUA_MODEL_ENCODING_CHARS_START_END+'.*', string=s)
            if (not m) or (len(m.groups())<1):
                raise Exception('Cannot find daehua encoding in "' + str(s) + '".')
            dh_encode_str = m.group(1)
            daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_ENCODE_STR] = su.StringUtils.trim(dh_encode_str)
            lg.Log.info(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Decoded mex part: ' + str(daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_ENCODE_STR])
            )

            #
            # Decode the mex pattern part and formula pattern part
            #
            # Split by '::'
            dh_objects_str = dh_encode_str.split(DaehuaModel.DAEHUA_MODEL_OBJECT_SEPARATOR)
            for dh_obj_str in dh_objects_str:
                # Break again
                parts = dh_obj_str.split(sep=DaehuaModel.DAEHUA_MODEL_OBJECT_DEFINITION_SYMBOL)
                parts = [su.StringUtils.trim(p) for p in parts]

                if parts[0] == DaehuaModel.DAEHUA_MODEL_OBJECT_VARS:
                    daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_OBJECT_VARS] = parts[1]
                elif parts[0] == DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER:
                    daehua_model_encoding_str[DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER] = parts[1]

            lg.Log.info(
                str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': Decoded mex pattern & formula pattern: ' + str(daehua_model_encoding_str)
            )
            return daehua_model_encoding_str
        except Exception as ex:
            errmsg = str(DaehuaModel.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                     + ': Failed to get encoding string for "' + str(s) + '". Exception ' + str(ex) + '.'
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
            raise Exception(errmsg)

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
        self.mex_obj = None

        self.daehua_model_str = DaehuaModel.get_daehua_model_encoding_str(
            s = self.encoding_str
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Model Encoding strings: ' + str(self.daehua_model_str)
        )
        try:
            self.mex_obj = mex.MatchExpression(
                pattern = self.daehua_model_str[DaehuaModel.DAEHUA_MODEL_OBJECT_VARS]
            )
        except Exception as ex_no_mex:
            error_msg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                       + ': Cannot compile mex pattern "'\
                       + str(self.daehua_model_str[DaehuaModel.DAEHUA_MODEL_OBJECT_VARS]) \
                       + '". Exception: ' + str(ex_no_mex) + '.'
            lg.Log.error(error_msg)
            raise Exception(error_msg)

        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Model Object vars: ' + str(self.mex_obj.mex_obj_vars)
        )
        return

    def get_answer(self):
        #
        # Extract variables from question
        #
        if self.mex_obj is None:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                + ': No mex object initialized'
            )

        var_values = self.mex_obj.get_params(
            return_one_value = True,
            sentence         = self.question
        )

        #
        # Extract formula from answer
        #
        formula_code_str = DaehuaModel.get_formula_code_str(
            daehua_answer_object_str = self.daehua_model_str[DaehuaModel.DAEHUA_MODEL_OBJECT_ANSWER],
            var_values = var_values
        )
        lg.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
            + ': Formula code string: "' + str(formula_code_str) + '".'
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
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_IMPORTANT

    tests = [
        {
            'encoding': 'Volume of Sphere -*-'
                        'vars=='
                        'r, float, radius / r   ;'
                        'd, float, diameter / d'
                        '::' + 'answer == (4/3) * (3.141592653589793 * $$r * $$r * $$r)'
                        '-*-',
            'questions': [
                'What is the volume of a sphere of radius 5.88?',
                'Radius is 5.88?',
            ]
        },
        {
            'encoding': '-*-'\
                   + 'vars==id, float, id / indo' \
                   + '::'\
                   + 'answer==('\
                   + '  ($$id -lt 0)*1*(1 + (1 | (-$$id)))'\
                   + '+ ($$id -gt 0)*1*(1 + $$id)'\
                   + ')-*-',
            'questions': [
                'What is -2.6 indo odds?',
                'What is +1.2 indo odds?',
                'Indo is 1.678'
            ]
        },
        {
            'encoding': '-*-'
                        'vars == '
                        'sendername, str-zh-cn, 】, right   ;'
                        'acc, number, 尾号 / 账号   ;'
                        'm, int, 月   ;   d, int, 日   ;   t, time, 完成   ;'
                        'amt, float, 民币 / 币   ;   '
                        'bal, float, 余额'
                        '::  answer == $$amt  -*-',
            'questions': [
                '【中国农业银行】 习近平 您尾号0579账户10月17日09:27完成代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】习近平 您尾号0579账户10月17日09:27:55完成代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】习近平 您尾号0579账户10月17日完成09:27代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】习近平 您尾号0579账户10月17日完成09:27:55代付交易人民币2309.95，余额2932.80。',
                '【中国农业银行】 习近平 您尾号 0579 账户 10月 17日 完成 09:27 代付交易 人民币 2309.95，余额 2932.80。',
                '【中国农业银行】 习近平 您尾号  0579 账户 10月 17日 完成 09:27:55 代付交易 人民币 2309.95，余额 2932.80。',
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
