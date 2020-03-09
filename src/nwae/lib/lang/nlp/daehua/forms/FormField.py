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
    KEY_MEX_VAR_NAME = 'mexVarName'
    KEY_MEX_VAR_TYPE = 'mexVarType'
    KEY_MEX_VAR_EXPRESSIONS = 'mexVarExpressions'
    KEY_COMPLETED = 'completed'
    
    @staticmethod
    def import_form_field(
            json_obj
    ):
        if_required = True
        if_masked = False
        mex_expr = None
        
        # Non-compulsory keys
        if FormField.KEY_IF_REQUIRED in json_obj.keys():
            if_required = json_obj[FormField.KEY_IF_REQUIRED]
        if FormField.KEY_IF_MASKED in json_obj.keys():
            if_masked = json_obj[FormField.KEY_IF_MASKED]
        if FormField.KEY_MEX_EXPR in json_obj.keys():
            mex_expr = json_obj[FormField.KEY_MEX_EXPR]

        return FormField(
            # Compulsory key
            name = json_obj[FormField.KEY_NAME],
            # Compulsory key
            value = json_obj[FormField.KEY_VALUE],
            if_required = if_required,
            if_masked   = if_masked,
            mex_expr    = mex_expr
        )

    def __init__(
            self,
            name,
            value,
            if_required,
            if_masked,
            # MEX expression to extract param from human sentence
            mex_expr
    ):
        self.name = name
        self.value = value
        self.if_required = if_required
        self.if_masked = if_masked
        # Field MEX
        self.mex_expr = mex_expr
        try:
            self.mex_obj = MatchExpression(
                pattern = self.mex_expr,
                lang    = None
            )
            self.mex_var_name = self.mex_obj.get_mex_var_names()[0]
            self.mex_var_type = self.mex_obj.get_mex_var_type(var_name=self.mex_var_name)
            self.mex_var_expressions = self.mex_obj.get_mex_var_expressions(var_name=self.mex_var_name)

            self.mex_obj_no_var_expressions = MatchExpression(
                pattern = self.mex_var_name + MatchExpression.MEX_VAR_DESCRIPTION_SEPARATOR
                          + self.mex_var_type + MatchExpression.MEX_VAR_DESCRIPTION_SEPARATOR
                          + ''
            )
        except Exception as ex_mex:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Failed to get mex var name for mex expr "' + str(self.mex_expr)
                + '", got exception "' + str(ex_mex) + '".'
            )
        # Already obtained the parameter from user conversation?
        self.completed = False
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Field initialized: ' + str(self.to_json())
         )
        return

    def set_field_value(
            self,
            user_text
    ):
        value = None
        # Try with var expressions first
        res = self.__set_field_value_from_text(
            text = user_text,
            exclude_var_expressions = False
        )
        if res is True:
            return True
        else:
            # Try to match with no text expressions, as user may just type the value alone
            res = self.__set_field_value_from_text(
                text = user_text,
                exclude_var_expressions = True
            )
            return res

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
            return True
        else:
            return False

    def to_json(self):
        return {
            FormField.KEY_NAME: self.name,
            FormField.KEY_VALUE: self.value,
            FormField.KEY_IF_REQUIRED: self.if_required,
            FormField.KEY_IF_MASKED: self.if_masked,
            FormField.KEY_MEX_EXPR: self.mex_expr,
            FormField.KEY_MEX_VAR_NAME: self.mex_var_name,
            FormField.KEY_MEX_VAR_TYPE: self.mex_var_type,
            FormField.KEY_MEX_VAR_EXPRESSIONS: self.mex_var_expressions,
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
        'mexExpr': 'amt,float,金额/amount'
    }
    ffld_obj = FormField.import_form_field(json_obj=fld)
    print(ffld_obj.to_json())

    text = 'the amount is 800.99'
    print(ffld_obj.set_field_value(user_text=text))
    print(ffld_obj.to_json())

    text = '777.88'
    print(ffld_obj.set_field_value(user_text=text))
    print(ffld_obj.to_json())