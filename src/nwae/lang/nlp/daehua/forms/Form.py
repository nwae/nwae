# -*- coding: utf-8 -*-

from nwae.utils.StringUtils import StringUtils
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.lang.nlp.daehua.forms.FormField as ffld
from mex.MatchExpression import MatchExpression


class Form:

    KEY_TITLE = 'title'
    # Message or instruction to user
    KEY_INSTRUCTION = 'instruction'
    KEY_IF_NEED_CONFIRM = 'ifNeedConfirm'
    KEY_FORM_FIELDS = 'fields'
    KEY_MEX_FORM_MODEL = 'mexFormModel'
    # For deserializing old objects so the old state is maintained
    KEY_FORM_COMPLETED = 'formCompleted'

    @staticmethod
    def import_form_fields(
            list_json,
            mex_form_model
    ):
        if len(list_json) != len(mex_form_model):
            raise Exception(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': List of fields must be same length with mex expr list.'
                + ' Fields: ' + str(list_json) + ', Mex Expr List: ' + str(mex_form_model)
            )
        form_fields = []
        for i in range(len(list_json)):
            json_field = list_json[i]
            json_field[ffld.FormField.KEY_MEX_EXPR] = StringUtils.trim(mex_form_model[i])
            try:
                form_fields.append(
                    ffld.FormField.import_form_field(json_obj=json_field)
                )
            except Exception as ex_field:
                errmsg = \
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Error importing field: ' + str(json_field) \
                    + '. Exception: ' + str(ex_field)
                Log.error(errmsg)
                raise Exception(errmsg)
        return form_fields

    @staticmethod
    def deserialize(
            form_json
    ):
        ffs = []
        for fld in form_json[Form.KEY_FORM_FIELDS]:
            ffs.append(ffld.FormField.deserialize(json_obj=fld))

        return Form(
            title           = form_json[Form.KEY_TITLE],
            instruction     = form_json[Form.KEY_INSTRUCTION],
            if_need_confirm = form_json[Form.KEY_IF_NEED_CONFIRM],
            form_fields     = ffs,
            mex_form_model  = form_json[Form.KEY_MEX_FORM_MODEL],
            form_completed  = form_json[Form.KEY_FORM_COMPLETED]
        )

    def __init__(
            self,
            title,
            instruction,
            if_need_confirm,
            # List of FormFields
            form_fields,
            # mex_form_model
            mex_form_model,
            # For deserializing old objects so the old state is mainted
            form_completed = False
    ):
        self.title = title
        self.instruction = instruction
        self.if_need_confirm = if_need_confirm
        # List of FormFields
        self.form_fields = form_fields
        # Field MEX
        self.mex_form_model = mex_form_model
        self.mex_obj = MatchExpression(
            pattern = self.mex_form_model,
            lang    = None,
        )
        self.form_completed = form_completed

    def get_title_text(self):
        return str(self.title)

    def get_instruction(self):
        return str(self.instruction)

    def reset_fields_to_incomplete(
            self
    ):
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Reset form fields to incomplete for form ' + str(self.to_json())
        )
        for i in range(len(self.form_fields)):
            fld = self.form_fields[i]
            fld.completed = False
            fld.value = None

    def reset_fields_value_just_updated(
            self
    ):
        for fld in self.form_fields:
            fld.reset_value_just_updated()

    def get_fields_values_text(
            self,
            text_newline_char,
            text_space_char,
            include_form_title = True
    ):
        text = ''
        if include_form_title:
            text = self.get_title_text() + text_newline_char

        for i in range(len(self.form_fields)):
            fld = self.form_fields[i]
            value_text = fld.value
            if value_text is None:
                value_text = '-'

            # Put bold text if just updated
            if fld.value_just_updated:
                value_text = '<b>' + str(value_text) + '</b>'

            text = text + text_space_char*2 + str(fld.name) \
                   + ': ' + str(value_text) + '' \
                   + text_newline_char
        return text

    def to_json_form_fields(self):
        ffs = []
        for fld in self.form_fields:
            ffs.append(fld.to_json())
        return ffs

    def to_json(self):
        return {
            Form.KEY_TITLE: self.title,
            Form.KEY_INSTRUCTION: self.instruction,
            Form.KEY_IF_NEED_CONFIRM: self.if_need_confirm,
            Form.KEY_FORM_FIELDS: self.to_json_form_fields(),
            Form.KEY_MEX_FORM_MODEL: self.mex_form_model,
            # For deserializing old objects so the old state is maintained
            Form.KEY_FORM_COMPLETED: self.form_completed
        }


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_IMPORTANT

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
    mex_form_model = 'name,str-en,name ; amt,float,金额/amount ; acc,account_number,账号/account'

    daehua_form = Form(
        title           = colform['text'],
        instruction     = colform['message'],
        if_need_confirm = colform['ifNeedConfirm'],
        form_fields     = Form.import_form_fields(list_json=colform['fields'], mex_form_model=mex_form_model.split(';')),
        mex_form_model  = mex_form_model
    )

    print(daehua_form.to_json())

    daehua_form_copy = Form.deserialize(
        form_json = daehua_form.to_json()
    )
    print(daehua_form_copy.to_json())
    # Must be True
    print(daehua_form_copy.to_json() == daehua_form.to_json())
    exit(0)
