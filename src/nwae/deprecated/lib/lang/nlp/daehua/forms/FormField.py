# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from mex.MatchExpression import MatchExpression


class FormField:

    KEY_NAME = 'name'
    KEY_VALUE = 'value'
    KEY_IF_REQUIRED = 'ifRequired'
    KEY_IF_MASKED = 'ifMasked'
    KEY_MEX_EXPR = 'mexExpr'
    # For deserializing old objects so the old state is maintained
    KEY_VALUE_JUST_UPDATED = 'valueJustUpdated'
    KEY_COMPLETED = 'completed'

    @staticmethod
    def deserialize(json_obj):
        return FormField.import_form_field(json_obj=json_obj)

    @staticmethod
    def import_form_field(
            json_obj
    ):
        if_required = True
        if_masked = False
        completed = False
        value_just_updated = False
        
        # Non-compulsory keys
        if FormField.KEY_IF_REQUIRED in json_obj.keys():
            if_required = json_obj[FormField.KEY_IF_REQUIRED]
        if FormField.KEY_IF_MASKED in json_obj.keys():
            if_masked = json_obj[FormField.KEY_IF_MASKED]
        if FormField.KEY_VALUE_JUST_UPDATED in json_obj.keys():
            value_just_updated = json_obj[FormField.KEY_VALUE_JUST_UPDATED]
        if FormField.KEY_COMPLETED in json_obj.keys():
            completed = json_obj[FormField.KEY_COMPLETED]

        return FormField(
            # Compulsory key
            name = json_obj[FormField.KEY_NAME],
            # Compulsory key
            value = json_obj[FormField.KEY_VALUE],
            if_required = if_required,
            if_masked   = if_masked,
            mex_expr    = json_obj[FormField.KEY_MEX_EXPR],
            value_just_updated = value_just_updated,
            completed   = completed
        )

    def __init__(
            self,
            name,
            value,
            if_required,
            if_masked,
            # MEX expression to extract param from human sentence
            mex_expr,
            # For deserializing old objects so the old state is maintained
            value_just_updated = False,
            completed = False
    ):
        self.name = name
        self.value = value
        self.if_required = if_required
        self.if_masked = if_masked
        # Field MEX
        self.mex_expr = mex_expr
        self.value_just_updated = value_just_updated
        # Already obtained the parameter from user conversation?
        self.completed = completed
        try:
            self.mex_obj = MatchExpression(
                pattern = self.mex_expr,
                lang    = None
            )
            self.mex_var_name = self.mex_obj.get_mex_var_names()[0]
            self.mex_obj_no_var_expressions = MatchExpression.create_mex_obj_from_object_vars(
                var_name_str = self.mex_var_name,
                var_type_str = self.mex_obj.get_mex_var_type(var_name=self.mex_var_name),
                var_expressions_str = '',
                var_len_range_list2 = self.mex_obj.get_mex_var_length_range(var_name=self.mex_var_name),
                var_preferred_dir_str = self.mex_obj.get_mex_var_pref_dir(var_name=self.mex_var_name)
            )
        except Exception as ex_mex:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Failed to get mex var name for mex expr "' + str(self.mex_expr)
                + '", got exception "' + str(ex_mex) + '".'
            )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Field initialized: ' + str(self.to_json())
         )
        return

    def set_value_just_updated(self):
        self.value_just_updated = True

    def reset_value_just_updated(self):
        self.value_just_updated = False

    def set_field_value(
            self,
            user_text,
            # Strict means match only with expressions
            strict_var_expressions = True
    ):
        value = None
        # Try with var expressions first
        res = self.__set_field_value_from_text(
            text = user_text,
            exclude_var_expressions = False
        )
        if res is True:
            return True
        elif not strict_var_expressions:
            # Try to match with no text expressions, as user may just type the value alone
            res = self.__set_field_value_from_text(
                text = user_text,
                exclude_var_expressions = True
            )
            return res
        else:
            return False

    def __set_field_value_from_text(
            self,
            text,
            exclude_var_expressions = False
    ):
        if exclude_var_expressions:
            params_dict = self.mex_obj_no_var_expressions.get_params(
                sentence = text,
                return_one_value = True
            )
        else:
            params_dict = self.mex_obj.get_params(
                sentence = text,
                # No need to return 2 sides
                return_one_value = True
            )
        if params_dict[self.mex_var_name] is not None:
            self.value = params_dict[self.mex_var_name]
            self.set_value_just_updated()
            self.completed = True
            return True
        else:
            return False

    # So that we can serialize state to file
    def to_json(self):
        return {
            FormField.KEY_NAME: self.name,
            FormField.KEY_VALUE: self.value,
            FormField.KEY_IF_REQUIRED: self.if_required,
            FormField.KEY_IF_MASKED: self.if_masked,
            FormField.KEY_MEX_EXPR: self.mex_expr,
            FormField.KEY_VALUE_JUST_UPDATED: self.value_just_updated,
            FormField.KEY_COMPLETED: self.completed
        }


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_INFO

    fld = {
        'name': 'Amount',
        'value': '',
        'type': 'text',
        'ifRequired': True,
        'ifMasked': True,
        'mexExpr': 'amt, float, 金额/amount, 1-6, right'
    }
    ffld_obj = FormField.import_form_field(json_obj=fld)
    print(ffld_obj.to_json())

    text = 'the amount is 800.99'
    print(ffld_obj.set_field_value(user_text=text))
    print(ffld_obj.to_json())

    text = '777.88'
    print(ffld_obj.set_field_value(user_text=text))
    print(ffld_obj.to_json())

    text = '55.11'
    # Should fail to set value, due to strict flag
    print(ffld_obj.set_field_value(user_text=text, strict_var_expressions=True))
    print(ffld_obj.to_json())

    # Serialize & Deserialize must be same
    ffld_copy = FormField.import_form_field(json_obj=ffld_obj.to_json())
    print(ffld_copy.to_json())
    # Must be True
    print(ffld_copy.to_json() == ffld_obj.to_json())