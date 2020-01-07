# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor

class SampleTextClassificationData:

    COL_CLASS_ID = 'classId'
    COL_CLASS = 'class'
    COL_CLASS_NAME = 'className'
    COL_TEXT = 'text'
    COL_TEXT_ID = 'textId'
    # Segmented text
    COL_TEXT_SEG = 'textSegmented'

    TYPE_IO_IN = 'in'
    TYPE_IO_OUT = 'out'

    # The class text
    SAMPLE_TRAINING_DATA = {
        LangFeatures.LANG_KO: {
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, '하나', '하나 두 두 셋 넷'),
                (1, '하나', '하나 하나 두 셋 셋 넷'),
                (1, '하나', '하나 두 셋 넷'),
                (2, '두', '두 셋 셋 넷'),
                (2, '두', '두 두 셋 셋 넷 넷'),
                (2, '두', '두 두 셋 넷 넷')
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, '하나', '하나 두 두 셋 넷', '하나 두 두 셋 넷'),
                (1, '하나', '하나 하나 두 셋 셋 넷', '하나 하나 두 셋 셋 넷'),
                (1, '하나', '하나 두 셋 넷', '하나 두 셋 넷'),
                # Appended intent name from processing
                (1, '하나', '하나', '하나'),
                (2, '두', '두 셋 셋 넷', '두 셋 셋 넷'),
                (2, '두', '두 두 셋 셋 넷 넷', '두 두 셋 셋 넷 넷'),
                (2, '두', '두 두 셋 넷 넷', '두 두 셋 넷 넷'),
                # Appended intent name from processing
                (2, '두', '두', '두'),
            )
        },
        LangFeatures.LANG_VN: {
            # Class/Intent ID, Class Name/Intent Name, Text
            TYPE_IO_IN: (
                (1, 'rút tiền', 'giới hạn rút tiền'),
                (1, 'rút tiền', 'rút bao nhiêu'),
                (1, 'rút tiền', 'trạng thái lệnh rút tiền')
            ),
            # Class/Intent ID, Class Name/Intent Name, Text, Text Segmented
            TYPE_IO_OUT: (
                (1, 'rút tiền', 'giới hạn rút tiền', 'giới hạn--||--rút tiền'),
                (1, 'rút tiền', 'rút bao nhiêu', 'rút--||--bao nhiêu'),
                (1, 'rút tiền', 'trạng thái lệnh rút tiền', 'trạng thái--||--lệnh--||--rút tiền'),
                # Appended intent name from processing
                (1, 'rút tiền', 'rút tiền', 'rút tiền'),
                # Appended latin equivalent forms
                (1, 'rút tiền', 'giới hạn rút tiền', 'gioi han--||--rut tien'),
                (1, 'rút tiền', 'rút bao nhiêu', 'rut--||--bao nhieu'),
                (1, 'rút tiền', 'trạng thái lệnh rút tiền', 'trang thai--||--lenh--||--rut tien'),
                (1, 'rút tiền', 'rút tiền', 'rut tien')
            )
        }
    }

    @staticmethod
    def get_text_classification_training_data(
            lang,
            type_io = TYPE_IO_IN
    ):
        sample_training_data = SampleTextClassificationData.SAMPLE_TRAINING_DATA[lang]
        class_arr = [y_x[0] for y_x in sample_training_data[type_io]]
        class_name_arr = [y_x[1] for y_x in sample_training_data[type_io]]
        texts_arr = [y_x[2] for y_x in sample_training_data[type_io]]
        texts_seg_arr = None
        if type_io == 'out':
            texts_seg_arr = [y_x[3] for y_x in sample_training_data[type_io]]

        return {
            SampleTextClassificationData.COL_CLASS_ID: range(1000,1000+len(class_arr),1),
            SampleTextClassificationData.COL_CLASS: class_arr,
            SampleTextClassificationData.COL_CLASS_NAME: class_name_arr,
            SampleTextClassificationData.COL_TEXT: texts_arr,
            SampleTextClassificationData.COL_TEXT_ID: range(2000,2000+len(class_arr),1),
            SampleTextClassificationData.COL_TEXT_SEG: texts_seg_arr
        }


if __name__ == '__main__':
    data = SampleTextClassificationData.get_text_classification_training_data(
        lang = LangFeatures.LANG_VN
    )
    df = pd.DataFrame(data)
    print(data)
    print(df)
