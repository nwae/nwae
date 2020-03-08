# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


class FormField:

    KEY_NAME = 'name'
    KEY_VALUE = 'value'
    KEY_IF_REQUIRED = 'ifRequired'
    KEY_IF_MASKED = 'ifMasked'
    KEY_MEX_EXPR = 'mexExpr'
    KEY_COMPLETED = 'completed'
    
    @staticmethod
    def import_form(
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
        # Already obtained the parameter from user conversation?
        self.completed = False

    def to_json(self):
        return {
            FormField.KEY_NAME: self.name,
            FormField.KEY_VALUE: self.value,
            FormField.KEY_IF_REQUIRED: self.if_required,
            FormField.KEY_IF_MASKED: self.if_masked,
            FormField.KEY_MEX_EXPR: self.mex_expr,
            FormField.KEY_COMPLETED: self.completed
        }


