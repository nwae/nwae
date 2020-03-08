# -*- coding: utf-8 -*-

from nwae.utils.StringUtils import StringUtils
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.lib.lang.nlp.daehua.forms.FormField as ffld


class Form:

    KEY_TITLE = 'title'
    # Message or instruction to user
    KEY_INSTRUCTION = 'instruction'
    KEY_IF_NEED_CONFIRM = 'ifNeedConfirm'
    KEY_FORM_FIELDS = 'fields'
    KEY_MEX_FORM_MODEL = 'mexFormModel'

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
                    ffld.FormField.import_form(json_obj=json_field)
                )
            except Exception as ex_field:
                errmsg = \
                    str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                    + ': Error importing field: ' + str(json_field) \
                    + '. Exception: ' + str(ex_field)
                Log.error(errmsg)
                raise Exception(errmsg)
        return form_fields

    def __init__(
            self,
            title,
            instruction,
            if_need_confirm,
            # List of FormFields
            form_fields,
            # mex_form_model
            mex_form_model
    ):
        self.title = title
        self.instruction = instruction
        self.if_need_confirm = if_need_confirm
        # List of FormFields
        self.form_fields = form_fields
        # Field MEX
        self.mex_form_model = mex_form_model
        self.form_completed = False

    def to_json(self):
        ffs = []
        for fld in self.form_fields:
            ffs.append(fld.to_json())

        return {
            Form.KEY_TITLE: self.title,
            Form.KEY_INSTRUCTION: self.instruction,
            Form.KEY_IF_NEED_CONFIRM: self.if_need_confirm,
            Form.KEY_FORM_FIELDS: ffs,
            Form.KEY_MEX_FORM_MODEL: self.mex_form_model
        }


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

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
    exit(0)
