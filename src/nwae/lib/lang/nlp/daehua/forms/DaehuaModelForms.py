# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from mex.MatchExpression import MatchExpression
import nwae.lib.lang.nlp.daehua.forms.Form as daehua_form
import nwae.lib.lang.nlp.daehua.forms.FormField as daehua_form_field
from nwae.utils.StringUtils import StringUtils


class DaehuaModelForms:

    DEFAULT_OK = ('y', 'ok', 'yes')
    def __init__(
            self,
            form,
            confirm_words = DEFAULT_OK,
            confirm_question = 'Please confirm answer ' + str(DEFAULT_OK)
    ):
        if type(form) is not daehua_form.Form:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong form type "' + str(type(form)) + '". Expected type "' + str(daehua_form.Form)
            )
        # Keep the original form, and extended params
        self.form = form
        self.confirm_words = confirm_words
        self.confirm_question = confirm_question

        self.mex_expressions = self.form.mex_form_model
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Mex Expressions: ' + str(self.mex_expressions) + '.'
        )

        self.reset()
        return

    def get_form(self):
        return self.form

    def reset(self):
        Log.important('Form reset')
        # The current field we are trying to extract from user
        self.conv_current_field_index = None
        self.conv_current_field_name = None
        self.conv_current_field = None
        # Field values all completed
        self.conv_completed = False
        # User already confirmed the form values
        self.conv_completed_and_confirmed = False
        # Reset fields
        self.form.reset_fields_to_incomplete()
        return

    def __set_conversation_complete(self, caller=None):
        Log.important('Conversation set to completed by ' + str(caller))
        self.conv_completed = True

    def __set_conversation_complete_and_confirmed(self, caller=None):
        Log.important('Conversation set to completed and confirmed by ' + str(caller))
        self.conv_completed_and_confirmed = True

    def get_next_question(
            self
    ):
        if self.conv_completed:
            return None

        self.conv_current_field_index = None
        self.conv_current_field_name = None
        self.conv_current_field = None

        # Find the next variable
        for i in range(len(self.form.form_fields)):
            fld = self.form.form_fields[i]
            if not fld.completed:
                self.conv_current_field_index = i
                self.conv_current_field_name = fld.name
                self.conv_current_field = fld
                break

        if self.conv_current_field_index is None:
            # Answer-Questioning completed
            self.__set_conversation_complete(caller='get_next_question')
            return None

        cur_field = self.form.form_fields[self.conv_current_field_index]
        question = 'Please provide ' + str(cur_field.name).lower() + '?'
        return question

    def extract_param_value(
            self,
            answer
    ):
        cur_field = self.form.form_fields[self.conv_current_field_index]

        (mex_var_name, mex_var_type, value) = self.__get_mex_params(
            answer   = answer,
            mex_expr = self.conv_current_field.mex_expr
        )
        if value is not None:
            self.conv_current_field.value = value
        else:
            mex_plain = str(mex_var_name) + ',' + str(mex_var_type) + ','
            (mex_var_name, mex_var_type, value) = self.__get_mex_params(
                answer   = answer,
                mex_expr = mex_plain
            )
            if value is not None:
                self.conv_current_field.value = value

        if value is not None:
            confirm_question = \
                str(cur_field.name).lower() + ': "' + str(value) + '"' \
                + '? ' + str(self.confirm_question)
            return (value, confirm_question)

        return (None, None)

    def __get_mex_params(
            self,
            answer,
            mex_expr
    ):
        answer = StringUtils.trim(answer)

        mex = MatchExpression(
            pattern = mex_expr,
            lang    = None
        )
        # Should have 1 variable only
        mex_var_name = mex.get_mex_var_names()[0]
        mex_var_type = mex.get_mex_var_type(var_name=mex_var_name)

        Log.important('Mex var name: ' + str(mex_var_name) + ', type: ' + str(mex_var_type))
        params_dict = mex.get_params(
            sentence = answer,
            # No need to return 2 sides
            return_one_value = True
        )
        # print(params_dict)
        if params_dict[mex_var_name] is not None:
            self.conv_current_field.value = params_dict[mex_var_name]
            return (mex_var_name, mex_var_type, params_dict[mex_var_name])
        else:
            return (mex_var_name, mex_var_type, None)

    def confirm_current_field(self):
        self.conv_current_field.completed = True

    def confirm_answer(
            self,
            answer
    ):
        answer = StringUtils.trim(answer)
        if answer in self.confirm_words:
            self.confirm_current_field()
            return True
        else:
            return False

    def confirm_form(
            self,
            answer
    ):
        answer = StringUtils.trim(answer)
        if answer in self.confirm_words:
            self.__set_conversation_complete_and_confirmed(caller='confirm_form')
            return True
        else:
            self.reset()
            return False

    def get_confirm_form_question(
            self):
        q = ''
        for field in self.get_completed_fields():
            q = q \
                + str(field[daehua_form_field.FormField.KEY_NAME]) \
                + ': ' + str(field[daehua_form_field.FormField.KEY_VALUE]) + '\n\r'
        q = q + self.confirm_question
        return q

    def get_completed_fields(self):
        completed_fields = []
        for i in range(len(self.form.form_fields)):
            fld = self.form.form_fields[i]
            if fld.completed:
                completed_fields.append(fld.to_json())
        return completed_fields


if __name__ == '__main__':
    colform = {
        'message': 'Please fill the form',
        'text': 'Cash Deposit',
        'ifNeedConfirm': False,
        'fields': [
            {'name': 'Name', 'value': '', 'type': 'text', 'ifRequired': False, 'ifMasked': True},
            {'name': 'Amount', 'value': '', 'type': 'text', 'ifRequired': True, 'ifMasked': True},
            {'name': 'Account No', 'value': '', 'type': 'text', 'ifRequired': False, 'ifMasked': True}
        ]
    }
    # Must be aligned with fields above
    mex_form_model = 'name,str-en,name/이름 ; amt,float,金额/amount ; acc,account_number,账号/account'

    dform = daehua_form.Form(
        title           = colform['text'],
        instruction     = colform['message'],
        if_need_confirm = colform['ifNeedConfirm'],
        form_fields     = daehua_form.Form.import_form_fields(list_json=colform['fields'], mex_form_model=mex_form_model.split(';')),
        mex_form_model  = mex_form_model
    )

    print(dform.to_json())

    fconv = DaehuaModelForms(
        form = dform
    )

    while not fconv.conv_completed_and_confirmed:
        print('Form complete status: ' + str(fconv.conv_completed))
        if fconv.conv_completed:
            print('Form values completed. Asking for confirmation..')
            answer = input(fconv.get_confirm_form_question())
            fconv.confirm_form(answer=answer)
            continue

        q = fconv.get_next_question()
        if q is None:
            continue
        answer = input(q + '\n\r')
        print('User answer: ' + str(answer))

        (value, confirm_question) = fconv.extract_param_value(
            answer = answer
        )
        if value is not None:
            fconv.confirm_current_field()
            #answer = input(confirm_question)
            #fconv.confirm_answer(answer=answer)
