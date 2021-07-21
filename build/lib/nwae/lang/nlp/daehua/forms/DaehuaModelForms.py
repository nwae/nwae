# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.lang.nlp.daehua.forms.Form as daehua_form
from nwae.utils.StringUtils import StringUtils


class retFieldsUpdated:
    def __init__(
            self,
            # At least 'name', 'value', 'confirmQuestion'
            dict_name_values
    ):
        self.field_name_values = dict_name_values
        self.is_updated = len(list(dict_name_values.keys())) > 0


class DaehuaModelForms:

    FORM_STATE_NONE = 'none'
    FORM_STATE_AWAIT_FIELD_VALUE = 'awaitFieldValue'
    FORM_STATE_AWAIT_FIELD_VALUE_CONFIRMATION = 'awaitFieldValueConfirmation'
    FORM_STATE_FORM_COMPLETED = 'formCompleted'
    FORM_STATE_AWAIT_FORM_CONFIRMATION = 'awaitFormConfirmation'
    FORM_STATE_FORM_COMPLETED_AND_CONFIRMED = 'formCompletedAndConfirmed'

    # For deserializing old objects so the old state is maintained
    KEY_FORM = 'form'
    KEY_TEXT_LIST_CONFIRM_WORDS = 'textListConfirmWords'
    KEY_TEXT_CONFIRM_QUESTION = 'textConfirmQuestion'
    KEY_TEXT_ASK_FIELD_VALUE_PREFIX = 'textAskFieldValuePrefix'
    KEY_TEXT_NEWLINE_CHAR = 'textNewlineChar'
    KEY_TEXT_SPACE_CHAR = 'textSpaceChar'
    KEY_TEXT_HTML_FONT_START_TAG = 'textHtmlFontStartTag'
    KEY_TEXT_HTML_FONT_END_TAG = 'textHtmlFontEndTag'
    KEY_ERROR_COUNT_QUIT_THRESHOLD = 'errorCountQuitThreshold'
    KEY_FORM_STATE = 'formState'
    KEY_FILL_FORM_CONTINUOUS_ERR_COUNT = 'fillFormContinuousErrCount'
    KEY_CONV_CURRENT_FIELD_INDEX = 'convCurrentFieldIndex'
    KEY_CONV_CURRENT_FIELD_NAME = 'convCurrentFieldName'

    DEFAULT_OK = ('y', 'ok', 'yes')

    @staticmethod
    def deserialize(json_obj):
        dh_form_obj = daehua_form.Form.deserialize(form_json=json_obj[DaehuaModelForms.KEY_FORM])

        obj = DaehuaModelForms(
            form = dh_form_obj,
            text_list_confirm_words = json_obj[DaehuaModelForms.KEY_TEXT_LIST_CONFIRM_WORDS],
            text_confirm_question = json_obj[DaehuaModelForms.KEY_TEXT_CONFIRM_QUESTION],
            text_ask_field_value_prefix = json_obj[DaehuaModelForms.KEY_TEXT_ASK_FIELD_VALUE_PREFIX],
            text_newline_char = json_obj[DaehuaModelForms.KEY_TEXT_NEWLINE_CHAR],
            text_space_char = json_obj[DaehuaModelForms.KEY_TEXT_SPACE_CHAR],
            text_html_font_start_tag = json_obj[DaehuaModelForms.KEY_TEXT_HTML_FONT_START_TAG],
            text_html_font_end_tag = json_obj[DaehuaModelForms.KEY_TEXT_HTML_FONT_END_TAG],
            error_count_quit_threshold = json_obj[DaehuaModelForms.KEY_ERROR_COUNT_QUIT_THRESHOLD],
            form_state = json_obj[DaehuaModelForms.KEY_FORM_STATE],
            fill_form_continuous_err_count = json_obj[DaehuaModelForms.KEY_FILL_FORM_CONTINUOUS_ERR_COUNT],
            conv_current_field_index = json_obj[DaehuaModelForms.KEY_CONV_CURRENT_FIELD_INDEX],
            conv_current_field_name = json_obj[DaehuaModelForms.KEY_CONV_CURRENT_FIELD_NAME]
        )
        return obj

    def __init__(
            self,
            form,
            text_list_confirm_words = DEFAULT_OK,
            text_confirm_question = 'Please confirm answer ' + str(DEFAULT_OK),
            text_ask_field_value_prefix = 'Please provide',
            text_newline_char = '<br/>',
            text_space_char = '&nbsp',
            text_html_font_start_tag = '<font color="blue">',
            text_html_font_end_tag = '</font>',
            # For deserializing old objects so the old state is maintained
            error_count_quit_threshold = 2,
            form_state = None,
            fill_form_continuous_err_count = 0,
            conv_current_field_index = None,
            conv_current_field_name = None
    ):
        if type(form) is not daehua_form.Form:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Wrong form type "' + str(type(form)) + '". Expected type "' + str(daehua_form.Form)
            )
        # Keep the original form, and extended params
        self.form = form
        self.text_list_confirm_words = [str(s) for s in text_list_confirm_words]
        self.text_confirm_question = str(text_confirm_question)
        self.text_ask_field_value_prefix = str(text_ask_field_value_prefix)
        self.text_newline_char = str(text_newline_char)
        self.text_space_char = str(text_space_char)
        self.text_html_font_start_tag = str(text_html_font_start_tag)
        self.text_html_font_end_tag = str(text_html_font_end_tag)
        self.error_count_quit_threshold = error_count_quit_threshold

        self.text_form_title = self.form.get_title_text()

        self.mex_expressions = self.form.mex_form_model
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Mex Expressions: ' + str(self.mex_expressions) + '.'
        )

        self.form_state = form_state
        self.fill_form_continuous_err_count = fill_form_continuous_err_count
        self.conv_current_field_index = conv_current_field_index
        self.conv_current_field_name = conv_current_field_name

        if self.form_state is None:
            self.reset()
        return

    def get_form(self):
        return self.form

    def get_form_state(self):
        return self.form_state

    def get_continous_error_count(self):
        return self.fill_form_continuous_err_count

    def set_state_none(self):
        self.form_state = DaehuaModelForms.FORM_STATE_NONE
    def is_state_none(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_NONE

    def set_state_await_field_value(self):
        self.form_state = DaehuaModelForms.FORM_STATE_AWAIT_FIELD_VALUE
    def is_state_await_field_value(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_AWAIT_FIELD_VALUE

    def set_state_await_field_value_confirmation(self):
        self.form_state = DaehuaModelForms.FORM_STATE_AWAIT_FIELD_VALUE_CONFIRMATION
    def is_state_await_field_value_confirmation(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_AWAIT_FIELD_VALUE_CONFIRMATION

    def set_state_form_completed(self):
        self.form_state = DaehuaModelForms.FORM_STATE_FORM_COMPLETED
    def is_state_form_completed(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_FORM_COMPLETED

    def set_state_await_form_confirmation(self):
        self.form_state = DaehuaModelForms.FORM_STATE_AWAIT_FORM_CONFIRMATION
    def is_state_await_form_confirmation(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_AWAIT_FORM_CONFIRMATION

    def set_state_form_completed_and_confirmed(self):
        self.form_state = DaehuaModelForms.FORM_STATE_FORM_COMPLETED_AND_CONFIRMED
    def is_state_form_completed_and_confirmed(self):
        return self.form_state == DaehuaModelForms.FORM_STATE_FORM_COMPLETED_AND_CONFIRMED

    def is_error_threshold_hit(self):
        return self.fill_form_continuous_err_count >= self.error_count_quit_threshold

    def increment_continuous_error_count(self):
        self.fill_form_continuous_err_count += 1

    def reset_continuous_error_count(self):
        self.fill_form_continuous_err_count = 0

    def reset(self):
        Log.important('Form reset')
        self.set_state_none()
        # The current field we are trying to extract from user
        self.conv_current_field_index = None
        self.conv_current_field_name = None
        # Previous field set by user
        # self.conv_completed_fields = []
        # Reset fields
        self.form.reset_fields_to_incomplete()
        return

    def get_next_question(
            self,
            include_all_fields_text = True
    ):
        if self.is_state_form_completed() or self.is_state_form_completed_and_confirmed():
            return None

        self.conv_current_field_index = None
        self.conv_current_field_name = None

        # Find the next variable
        for i in range(len(self.form.form_fields)):
            fld = self.form.form_fields[i]
            if not fld.completed:
                self.conv_current_field_index = i
                self.conv_current_field_name = fld.name
                break

        if self.conv_current_field_index is None:
            # Answer-Questioning completed
            self.set_state_form_completed()
            return None

        cur_field = self.form.form_fields[self.conv_current_field_index]

        if include_all_fields_text:
            header_text = self.form.get_fields_values_text(
                text_newline_char  = self.text_newline_char,
                text_space_char    = self.text_space_char,
                include_form_title = True
            )
        else:
            header_text = self.text_form_title + self.text_newline_char

        question = self.text_html_font_start_tag \
                   + header_text
        if self.get_continous_error_count() > 0:
            question = question \
                       + '(Try ' + str(self.get_continous_error_count() + 1) \
                       + ')' + str(self.text_space_char)
        question = question \
                   + self.text_ask_field_value_prefix \
                   + ' ' + str(cur_field.name).lower() + '?'\
                   + self.text_html_font_end_tag

        self.form.reset_fields_value_just_updated()

        return question

    def try_to_update_fields(
            self,
            answer,
            update_current_then_all = False
    ):
        try_count = 1
        result = retFieldsUpdated(dict_name_values={})

        if update_current_then_all:
            result = self.set_current_field_value_from_answer(
                answer = answer,
                strict_expressions = True
            )
            Log.info(
                'Try update ' + str(try_count)
                + ': Strict update on current field "' + self.conv_current_field_name
                + '", updated = ' + str(result.is_updated)
            )
            try_count += 1
            if result.is_updated:
                self.confirm_current_field()
                # answer = input(confirm_question)
                # fconv.confirm_answer(answer=answer)

        if not result.is_updated:
            # Try to update all
            result = self.set_all_field_value_from_answer(
                answer = answer
            )
            Log.info(
                'Try update ' + str(try_count) + ': Strict update on all fields'
                + ', updated = ' + str(result.is_updated)
            )
            try_count += 1

            if not result.is_updated:
                result = self.set_current_field_value_from_answer(
                    answer = answer,
                    strict_expressions = False
                )
                Log.info(
                    'Try update 3: Not strict update on current field "' + self.conv_current_field_name
                    + '", updated = ' + str(result.is_updated)
                )
        if result.is_updated:
            self.reset_continuous_error_count()
        else:
            self.increment_continuous_error_count()
        return result

    def set_all_field_value_from_answer(
            self,
            answer
    ):
        dict_fld_name_values_updated = {}
        for fld in self.form.form_fields:
            value = self.__set_field_value_from_answer(
                answer = answer,
                form_field = fld,
                # For setting not targeted field, make sure it is strict
                strict_var_expressions = True
            )
            if value is not None:
                Log.info('********* Field "' + str(fld.name) + '" updated value = ' + str(value))
                dict_fld_name_values_updated[fld.name] = fld.name

        return retFieldsUpdated(
            dict_name_values = dict_fld_name_values_updated
        )

    def set_current_field_value_from_answer(
            self,
            answer,
            strict_expressions
    ):
        value = self.__set_field_value_from_answer(
            answer = answer,
            form_field = self.form.form_fields[self.conv_current_field_index],
            strict_var_expressions = strict_expressions
        )
        if value is None:
            return retFieldsUpdated(dict_name_values={})
        else:
            return retFieldsUpdated(
                dict_name_values = {self.conv_current_field_name: value}
            )

    def __set_field_value_from_answer(
            self,
            answer,
            form_field,
            strict_var_expressions
    ):
        res = form_field.set_field_value(
            user_text = answer,
            # Allow to match also single word or number (e.g. "79.5"),
            # without any var expressions (with expressions, "Amount is 79.5")
            strict_var_expressions = strict_var_expressions
        )
        Log.info(
            'Updated field "' + str(form_field.name)
            + '" = ' + str(res)
        )
        if res is True:
            # Confirm question we can build elsewhere
            # confirm_question = \
            #     str(form_field.name).lower() + ': "' + str(value) + '"' \
            #     + '? ' + str(self.text_confirm_question)
            return form_field.value

        return None

    def confirm_current_field(self):
        self.form.form_fields[self.conv_current_field_index].completed = True

    def confirm_answer(
            self,
            answer
    ):
        answer = StringUtils.trim(answer)
        if answer.lower() in self.text_list_confirm_words:
            self.confirm_current_field()
            return True
        else:
            # No form confirmation
            return False

    def confirm_form(
            self,
            answer
    ):
        answer = StringUtils.trim(answer)
        if answer.lower() in self.text_list_confirm_words:
            self.set_state_form_completed_and_confirmed()
            self.reset_continuous_error_count()
            return True
        else:
            # Try to update all fields strictly, maybe user wants to change something
            result = self.set_all_field_value_from_answer(
                answer = answer
            )
            if result.is_updated:
                self.reset_continuous_error_count()
            else:
                self.increment_continuous_error_count()

            if self.is_error_threshold_hit():
                Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Reset form after ' + str(self.fill_form_continuous_err_count) + ' error counts.'
                )
                self.reset()
            # No form confirmation
            return False

    def get_confirm_form_question(
            self):
        final_form_text = self.form.get_fields_values_text(
            text_newline_char = self.text_newline_char,
            text_space_char   = self.text_space_char,
            include_form_title = True
        )
        q = self.text_html_font_start_tag + final_form_text

        if self.get_continous_error_count() > 0:
            q = q \
                + '(Try ' + str(self.get_continous_error_count() + 1) \
                + ')' + str(self.text_space_char)

        q = q \
            + self.text_confirm_question \
            + self.text_html_font_end_tag

        self.form.reset_fields_value_just_updated()

        return q

    def get_completed_fields(self):
        completed_fields = []
        for i in range(len(self.form.form_fields)):
            fld = self.form.form_fields[i]
            if fld.completed:
                completed_fields.append(fld.to_json())
        return completed_fields

    def simulate_question_answer(
            self
    ):
        while not self.is_state_form_completed_and_confirmed():
            if self.is_error_threshold_hit():
                print('User quits form conversation.')
                break
            q = self.get_next_question()
            if q is None:
                self.set_state_form_completed()
                print('Form values completed. Asking for confirmation..')
                self.set_state_await_form_confirmation()

                answer = input(self.get_confirm_form_question())
                self.confirm_form(answer=answer)
                continue
            else:
                self.set_state_await_field_value()

                answer = input(q + '\n\r')
                print('User answer: ' + str(answer))

                result = self.try_to_update_fields(answer=answer)
                print('***** Updated fields: ' + str(result.field_name_values))

    def to_json(self):
        return {
            DaehuaModelForms.KEY_FORM: self.form.to_json(),
            DaehuaModelForms.KEY_TEXT_LIST_CONFIRM_WORDS: self.text_list_confirm_words,
            DaehuaModelForms.KEY_TEXT_CONFIRM_QUESTION: self.text_confirm_question,
            DaehuaModelForms.KEY_TEXT_ASK_FIELD_VALUE_PREFIX: self.text_ask_field_value_prefix,
            DaehuaModelForms.KEY_TEXT_NEWLINE_CHAR: self.text_newline_char,
            DaehuaModelForms.KEY_TEXT_SPACE_CHAR: self.text_space_char,
            DaehuaModelForms.KEY_TEXT_HTML_FONT_START_TAG: self.text_html_font_start_tag,
            DaehuaModelForms.KEY_TEXT_HTML_FONT_END_TAG: self.text_html_font_end_tag,
            DaehuaModelForms.KEY_ERROR_COUNT_QUIT_THRESHOLD: self.error_count_quit_threshold,
            DaehuaModelForms.KEY_FORM_STATE: self.form_state,
            DaehuaModelForms.KEY_FILL_FORM_CONTINUOUS_ERR_COUNT: self.fill_form_continuous_err_count,
            DaehuaModelForms.KEY_CONV_CURRENT_FIELD_INDEX: self.conv_current_field_index,
            DaehuaModelForms.KEY_CONV_CURRENT_FIELD_NAME: self.conv_current_field_name
        }


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
    mex_form_model = 'name,str,name/叫/이름,2-3,right ; amt,float,金额/amount ; acc,account_number,账号/account'

    dform = daehua_form.Form(
        title           = colform['text'],
        instruction     = colform['message'],
        if_need_confirm = colform['ifNeedConfirm'],
        form_fields     = daehua_form.Form.import_form_fields(list_json=colform['fields'], mex_form_model=mex_form_model.split(';')),
        mex_form_model  = mex_form_model
    )

    print(dform.to_json())

    daehua_model_forms = DaehuaModelForms(
        form = dform
    )
    print('************************************************')
    print(daehua_model_forms.to_json())
    print('************************************************')
    daehua_model_forms_copy = DaehuaModelForms.deserialize(json_obj = daehua_model_forms.to_json())
    print(daehua_model_forms_copy.to_json())
    print(daehua_model_forms_copy.to_json() == daehua_model_forms.to_json())

    daehua_model_forms.simulate_question_answer()
    print(daehua_model_forms.to_json())