# -*- coding: utf-8 -*-

import re
import mozg.common.util.Log as lg


#
# У бота цель которая требуется извлечь параметры
#
# Задачи и Идея:
#   1. По ПО, решение эты задачи достаточно через множество регехов
#      E.g. '.*(key)[:;= ]+["\']*([a-zA-Z]+).*', '.*(key)[:;= ]+["\']*([0-9]+).*'
#   2. По ИИ, как моделировать?
#
class DialogueGetValues:

    TYPE_ALL_REGEX =\
        '[.。,' +\
        '0-9a-zA-Z' +\
        '\u4e00-\u9fff' +\
        '\u0e01-\u0e5b' +\
        ']+'

    TYPE_STRING_NAME = 'string'
    # Handle for English, Chinese, Thai
    TYPE_STRING_REGEX =\
        '[0-9a-zA-Z' +\
        '\u4e00-\u9fff' +\
        '\u0e01-\u0e5b' +\
        ']+'
    TYPE_NUMBER_NAME = 'number'
    TYPE_NUMBER_REGEX = '[0-9]+[0-9,]*[.。]{0,1}[0-9]*'

    def __init__(self):
        return

    #
    # Сдали совокупность ключ-типы, постараемся извлечь значения ключей
    # через разговор.
    #
    def get_values(self, key_types):

        lg.Log.log('Starting conversation to retrieve the following:')
        lg.Log.log(key_types)

        key_values = {}
        missing_keys = list(key_types.keys())
        # Initialize
        for key in list(key_types.keys()):
            key_values[key] = None

        is_completed = False
        while not is_completed:

            #
            # Find filled keys
            #
            filled_keys = []
            for key in list(key_values.keys()):
                if not key_values[key] is None:
                    filled_keys = filled_keys + [key]

            dialogue = ''
            if len(filled_keys) > 0:
                dialogue = 'The following values have been provided:\r\n'
                for key in filled_keys:
                    dialogue = dialogue + '\t' + key + ': "' + key_values[key] + '"\r\n'

            dialogue = dialogue + 'Please provide: ' + str(missing_keys) + ": "
            kv_input = input(dialogue)

            for key in list(key_values.keys()):
                # Get type to expect
                kt = key_types[key]

                # Match for all strings/numbers/etc first
                pat = None
                if kt == DialogueGetValues.TYPE_NUMBER_NAME:
                    pat = '.*(' + key + ')[:;= 是]*["\']*(' + DialogueGetValues.TYPE_NUMBER_REGEX + ').*'
                elif kt == DialogueGetValues.TYPE_STRING_NAME:
                    pat = '.*(' + key + ')[:;= 是]*["\']*(' + DialogueGetValues.TYPE_STRING_REGEX + ').*'
                else:
                    raise Exception(str(self.__class__) + 'Unrecognized type ' + kt)

                if re.match(pattern=pat, string=kv_input):
                    # lg.Log.log('Input [' + kv_input + '] matched with pattern [' + pat + ']')
                    val = re.sub(pattern=pat, repl='\\2', string=kv_input)

                    if key_values[key] is None:
                        lg.Log.log('Got ' + key + ' as ' + val + '...')
                    else:
                        lg.Log.log('Replacing ' + key + ' (old value=' + key_values[key] + ') with ' + val + '.')
                    key_values[key] = val
                #else:
                    #lg.Log.log('Input [' + kv_input + '] NOT matched with pattern [' + pat + ']')

            #
            # Find those keys where value is still None
            #
            missing_keys = []
            for key in list(key_values.keys()):
                if key_values[key] is None:
                    missing_keys = missing_keys + [key]

            is_completed = len(missing_keys) == 0

            if is_completed:
                confirmation = input('Confirm the following values:' + str(key_values))
                if confirmation == 'ok':
                    break
                else:
                    is_completed = False

        return key_values


if __name__ == '__main__':
    dl = DialogueGetValues()
    key_types = {
        '名字': DialogueGetValues.TYPE_STRING_NAME,
        '岁': DialogueGetValues.TYPE_NUMBER_NAME,
        'ชื่': DialogueGetValues.TYPE_STRING_NAME
    }
    kv = dl.get_values(key_types=key_types)
    print(kv)