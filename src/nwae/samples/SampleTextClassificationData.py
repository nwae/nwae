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

    # The class text
    SAMPLE_TRAINING_DATA = {
        LangFeatures.LANG_KO: (
            # Class, Class Name, Text, Text Segmented, Training Data ID
            (1, '하나', '하나 두 두 셋 넷'),
            (1, '하나', '하나 하나 두 셋 셋 넷'),
            (1, '하나', '하나 두 셋 넷'),
            (2, '두', '두 셋 셋 넷'),
            (2, '두', '두 두 셋 셋 넷 넷'),
            (2, '두', '두 두 셋 넷 넷'),
            (3, '넷', '넷 다섯 다섯 여섯 여섯 여섯'),
            (3, '넷', '두 넷 넷 다섯 다섯 여섯 여섯'),
            (3, '넷', '두 넷 다섯 여섯 여섯'),
            (4, '여섯', '하나 여섯'),
            (4, '여섯', '하나 여섯 여섯'),
            (4, '여섯', '하나 하나 여섯'),
            (4, '여섯', '두 셋 넷 다섯'),
            (4, '여섯', '두 셋 셋 넷 다섯'),
            (4, '여섯', '두 셋 넷 넷 다섯')
        ),
        LangFeatures.LANG_VN: (
            (1, 'rút tiền', 'giới hạn rút tiền', 'giới hạn--||--rút tiền'),
            (1, 'rút tiền', 'rút bao nhiêu', 'rút--||--bao nhiêu'),
            (1, 'rút tiền', 'trạng thái lệnh rút tiền', 'trạng thái--||--lệnh--||--rút tiền')
        )
    }

    @staticmethod
    def get_text_classification_training_data(
            lang
    ):
        sample_training_data = SampleTextClassificationData.SAMPLE_TRAINING_DATA[lang]
        class_arr = [y_x[0] for y_x in sample_training_data]
        class_name_arr = [y_x[1] for y_x in sample_training_data]
        texts_arr = [y_x[2] for y_x in sample_training_data]

        return {
            SampleTextClassificationData.COL_CLASS_ID: range(1000,1000+len(class_arr),1),
            SampleTextClassificationData.COL_CLASS: class_arr,
            SampleTextClassificationData.COL_CLASS_NAME: class_name_arr,
            SampleTextClassificationData.COL_TEXT: texts_arr,
            SampleTextClassificationData.COL_TEXT_ID: range(2000,2000+len(class_arr),1)
        }


if __name__ == '__main__':
    data = SampleTextClassificationData.get_text_classification_training_data(
        lang = LangFeatures.LANG_VN
    )
    df = pd.DataFrame(data)
    print(data)
    print(df)
